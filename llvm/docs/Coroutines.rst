=====================================
Coroutines in LLVM
=====================================

.. contents::
   :local:
   :depth: 3

.. warning::
  This is a work in progress. Compatibility across LLVM releases is not 
  guaranteed.

Introduction
============

.. _coroutine handle:

LLVM coroutines are functions that have one or more `suspend points`_. 
When a suspend point is reached, the execution of a coroutine is suspended and
control is returned back to its caller. A suspended coroutine can be resumed 
to continue execution from the last suspend point or it can be destroyed. 

In the following example, we call function `f` (which may or may not be a 
coroutine itself) that returns a handle to a suspended coroutine 
(**coroutine handle**) that is used by `main` to resume the coroutine twice and
then destroy it:

.. code-block:: llvm

  define i32 @main() {
  entry:
    %hdl = call i8* @f(i32 4)
    call void @llvm.coro.resume(i8* %hdl)
    call void @llvm.coro.resume(i8* %hdl)
    call void @llvm.coro.destroy(i8* %hdl)
    ret i32 0
  }

.. _coroutine frame:

In addition to the function stack frame which exists when a coroutine is 
executing, there is an additional region of storage that contains objects that 
keep the coroutine state when a coroutine is suspended. This region of storage
is called the **coroutine frame**. It is created when a coroutine is called
and destroyed when a coroutine either runs to completion or is destroyed
while suspended.

LLVM currently supports two styles of coroutine lowering. These styles
support substantially different sets of features, have substantially
different ABIs, and expect substantially different patterns of frontend
code generation. However, the styles also have a great deal in common.

In all cases, an LLVM coroutine is initially represented as an ordinary LLVM
function that has calls to `coroutine intrinsics`_ defining the structure of
the coroutine. The coroutine function is then, in the most general case,
rewritten by the coroutine lowering passes to become the "ramp function",
the initial entrypoint of the coroutine, which executes until a suspend point
is first reached. The remainder of the original coroutine function is split
out into some number of "resume functions". Any state which must persist
across suspensions is stored in the coroutine frame. The resume functions
must somehow be able to handle either a "normal" resumption, which continues
the normal execution of the coroutine, or an "abnormal" resumption, which
must unwind the coroutine without attempting to suspend it.

Switched-Resume Lowering
------------------------

In LLVM's standard switched-resume lowering, signaled by the use of
`llvm.coro.id`, the coroutine frame is stored as part of a "coroutine
object" which represents a handle to a particular invocation of the
coroutine.  All coroutine objects support a common ABI allowing certain
features to be used without knowing anything about the coroutine's
implementation:

- A coroutine object can be queried to see if it has reached completion
  with `llvm.coro.done`.

- A coroutine object can be resumed normally if it has not already reached
  completion with `llvm.coro.resume`.

- A coroutine object can be destroyed, invalidating the coroutine object,
  with `llvm.coro.destroy`.  This must be done separately even if the
  coroutine has reached completion normally.

- "Promise" storage, which is known to have a certain size and alignment,
  can be projected out of the coroutine object with `llvm.coro.promise`.
  The coroutine implementation must have been compiled to define a promise
  of the same size and alignment.

In general, interacting with a coroutine object in any of these ways while
it is running has undefined behavior.

The coroutine function is split into three functions, representing three
different ways that control can enter the coroutine:

1. the ramp function that is initially invoked, which takes arbitrary
   arguments and returns a pointer to the coroutine object;

2. a coroutine resume function that is invoked when the coroutine is resumed,
   which takes a pointer to the coroutine object and returns `void`;

3. a coroutine destroy function that is invoked when the coroutine is
   destroyed, which takes a pointer to the coroutine object and returns
   `void`.

Because the resume and destroy functions are shared across all suspend
points, suspend points must store the index of the active suspend in
the coroutine object, and the resume/destroy functions must switch over
that index to get back to the correct point.  Hence the name of this
lowering.

Pointers to the resume and destroy functions are stored in the coroutine
object at known offsets which are fixed for all coroutines.  A completed
coroutine is represented with a null resume function.

There is a somewhat complex protocol of intrinsics for allocating and
deallocating the coroutine object.  It is complex in order to allow the
allocation to be elided due to inlining.  This protocol is discussed
in further detail below.

The frontend may generate code to call the coroutine function directly;
this will become a call to the ramp function and will return a pointer
to the coroutine object.  The frontend should always resume or destroy
the coroutine using the corresponding intrinsics.

Returned-Continuation Lowering
------------------------------

In returned-continuation lowering, signaled by the use of
`llvm.coro.id.retcon` or `llvm.coro.id.retcon.once`, some aspects of
the ABI must be handled more explicitly by the frontend.

In this lowering, every suspend point takes a list of "yielded values"
which are returned back to the caller along with a function pointer,
called the continuation function.  The coroutine is resumed by simply
calling this continuation function pointer.  The original coroutine
is divided into the ramp function and then an arbitrary number of
these continuation functions, one for each suspend point.

LLVM actually supports two closely-related returned-continuation
lowerings:

- In normal returned-continuation lowering, the coroutine may suspend
  itself multiple times. This means that a continuation function
  itself returns another continuation pointer, as well as a list of
  yielded values.

  The coroutine indicates that it has run to completion by returning
  a null continuation pointer. Any yielded values will be `undef`
  should be ignored.

- In yield-once returned-continuation lowering, the coroutine must
  suspend itself exactly once (or throw an exception).  The ramp
  function returns a continuation function pointer and yielded
  values, but the continuation function simply returns `void`
  when the coroutine has run to completion.

The coroutine frame is maintained in a fixed-size buffer that is
passed to the `coro.id` intrinsic, which guarantees a certain size
and alignment statically. The same buffer must be passed to the
continuation function(s). The coroutine will allocate memory if the
buffer is insufficient, in which case it will need to store at
least that pointer in the buffer; therefore the buffer must always
be at least pointer-sized. How the coroutine uses the buffer may
vary between suspend points.

In addition to the buffer pointer, continuation functions take an
argument indicating whether the coroutine is being resumed normally
(zero) or abnormally (non-zero).

LLVM is currently ineffective at statically eliminating allocations
after fully inlining returned-continuation coroutines into a caller.
This may be acceptable if LLVM's coroutine support is primarily being
used for low-level lowering and inlining is expected to be applied
earlier in the pipeline.

Async Lowering
--------------

In async-continuation lowering, signaled by the use of `llvm.coro.id.async`,
handling of control-flow must be handled explicitly by the frontend.

In this lowering, a coroutine is assumed to take the current `async context` as
one of its arguments (the argument position is determined by
`llvm.coro.id.async`). It is used to marshal arguments and return values of the
coroutine. Therefore an async coroutine returns `void`.

.. code-block:: llvm

  define swiftcc void @async_coroutine(i8* %async.ctxt, i8*, i8*) {
  }

Values live across a suspend point need to be stored in the coroutine frame to
be available in the continuation function. This frame is stored as a tail to the
`async context`.

Every suspend point takes an `context projection function` argument which
describes how-to obtain the continuations `async context` and every suspend
point has an associated `resume function` denoted by the
`llvm.coro.async.resume` intrinsic. The coroutine is resumed by calling this
`resume function` passing the `async context` as the one of its arguments
argument. The `resume function` can restore its (the caller's) `async context`
by applying a `context projection function` that is provided by the frontend as
a parameter to the `llvm.coro.suspend.async` intrinsic.

.. code-block:: c

  // For example:
  struct async_context {
    struct async_context *caller_context;
    ...
  }

  char *context_projection_function(struct async_context *callee_ctxt) {
     return callee_ctxt->caller_context;
  }

.. code-block:: llvm

  %resume_func_ptr = call i8* @llvm.coro.async.resume()
  call {i8*, i8*, i8*} (i8*, i8*, ...) @llvm.coro.suspend.async(
                                              i8* %resume_func_ptr,
                                              i8* %context_projection_function

The frontend should provide a `async function pointer` struct associated with
each async coroutine by `llvm.coro.id.async`'s argument. The initial size and
alignment of the `async context` must be provided as arguments to the
`llvm.coro.id.async` intrinsic. Lowering will update the size entry with the
coroutine frame  requirements. The frontend is responsible for allocating the
memory for the `async context` but can use the `async function pointer` struct
to obtain the required size.

.. code-block:: c

  struct async_function_pointer {
    uint32_t relative_function_pointer_to_async_impl;
    uint32_t context_size;
  }

Lowering will split an async coroutine into a ramp function and one resume
function per suspend point.

How control-flow is passed between caller, suspension point, and back to
resume function is left up to the frontend.

The suspend point takes a function and its arguments. The function is intended
to model the transfer to the callee function. It will be tail called by
lowering and therefore must have the same signature and calling convention as
the async coroutine.

.. code-block:: llvm

  call {i8*, i8*, i8*} (i8*, i8*, ...) @llvm.coro.suspend.async(
                   i8* %resume_func_ptr,
                   i8* %context_projection_function,
                   i8* (bitcast void (i8*, i8*, i8*)* to i8*) %suspend_function,
                   i8* %arg1, i8* %arg2, i8 %arg3)

Coroutines by Example
=====================

The examples below are all of switched-resume coroutines.

Coroutine Representation
------------------------

Let's look at an example of an LLVM coroutine with the behavior sketched
by the following pseudo-code.

.. code-block:: c++

  void *f(int n) {
     for(;;) {
       print(n++);
       <suspend> // returns a coroutine handle on first suspend
     }     
  } 

This coroutine calls some function `print` with value `n` as an argument and
suspends execution. Every time this coroutine resumes, it calls `print` again with an argument one bigger than the last time. This coroutine never completes by itself and must be destroyed explicitly. If we use this coroutine with 
a `main` shown in the previous section. It will call `print` with values 4, 5 
and 6 after which the coroutine will be destroyed.

The LLVM IR for this coroutine looks like this:

.. code-block:: llvm

  define i8* @f(i32 %n) {
  entry:
    %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
    %size = call i32 @llvm.coro.size.i32()
    %alloc = call i8* @malloc(i32 %size)
    %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %alloc)
    br label %loop
  loop:
    %n.val = phi i32 [ %n, %entry ], [ %inc, %loop ]
    %inc = add nsw i32 %n.val, 1
    call void @print(i32 %n.val)
    %0 = call i8 @llvm.coro.suspend(token none, i1 false)
    switch i8 %0, label %suspend [i8 0, label %loop
                                  i8 1, label %cleanup]
  cleanup:
    %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
    call void @free(i8* %mem)
    br label %suspend
  suspend:
    %unused = call i1 @llvm.coro.end(i8* %hdl, i1 false)
    ret i8* %hdl
  }

The `entry` block establishes the coroutine frame. The `coro.size`_ intrinsic is
lowered to a constant representing the size required for the coroutine frame. 
The `coro.begin`_ intrinsic initializes the coroutine frame and returns the 
coroutine handle. The second parameter of `coro.begin` is given a block of memory 
to be used if the coroutine frame needs to be allocated dynamically.
The `coro.id`_ intrinsic serves as coroutine identity useful in cases when the
`coro.begin`_ intrinsic get duplicated by optimization passes such as 
jump-threading.

The `cleanup` block destroys the coroutine frame. The `coro.free`_ intrinsic, 
given the coroutine handle, returns a pointer of the memory block to be freed or
`null` if the coroutine frame was not allocated dynamically. The `cleanup` 
block is entered when coroutine runs to completion by itself or destroyed via
call to the `coro.destroy`_ intrinsic.

The `suspend` block contains code to be executed when coroutine runs to 
completion or suspended. The `coro.end`_ intrinsic marks the point where 
a coroutine needs to return control back to the caller if it is not an initial 
invocation of the coroutine. 

The `loop` blocks represents the body of the coroutine. The `coro.suspend`_ 
intrinsic in combination with the following switch indicates what happens to 
control flow when a coroutine is suspended (default case), resumed (case 0) or 
destroyed (case 1).

Coroutine Transformation
------------------------

One of the steps of coroutine lowering is building the coroutine frame. The
def-use chains are analyzed to determine which objects need be kept alive across
suspend points. In the coroutine shown in the previous section, use of virtual register 
`%inc` is separated from the definition by a suspend point, therefore, it 
cannot reside on the stack frame since the latter goes away once the coroutine 
is suspended and control is returned back to the caller. An i32 slot is 
allocated in the coroutine frame and `%inc` is spilled and reloaded from that
slot as needed.

We also store addresses of the resume and destroy functions so that the 
`coro.resume` and `coro.destroy` intrinsics can resume and destroy the coroutine
when its identity cannot be determined statically at compile time. For our 
example, the coroutine frame will be:

.. code-block:: llvm

  %f.frame = type { void (%f.frame*)*, void (%f.frame*)*, i32 }

After resume and destroy parts are outlined, function `f` will contain only the 
code responsible for creation and initialization of the coroutine frame and 
execution of the coroutine until a suspend point is reached:

.. code-block:: llvm

  define i8* @f(i32 %n) {
  entry:
    %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
    %alloc = call noalias i8* @malloc(i32 24)
    %0 = call noalias i8* @llvm.coro.begin(token %id, i8* %alloc)
    %frame = bitcast i8* %0 to %f.frame*
    %1 = getelementptr %f.frame, %f.frame* %frame, i32 0, i32 0
    store void (%f.frame*)* @f.resume, void (%f.frame*)** %1
    %2 = getelementptr %f.frame, %f.frame* %frame, i32 0, i32 1
    store void (%f.frame*)* @f.destroy, void (%f.frame*)** %2
   
    %inc = add nsw i32 %n, 1
    %inc.spill.addr = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 2
    store i32 %inc, i32* %inc.spill.addr
    call void @print(i32 %n)
   
    ret i8* %frame
  }

Outlined resume part of the coroutine will reside in function `f.resume`:

.. code-block:: llvm

  define internal fastcc void @f.resume(%f.frame* %frame.ptr.resume) {
  entry:
    %inc.spill.addr = getelementptr %f.frame, %f.frame* %frame.ptr.resume, i64 0, i32 2
    %inc.spill = load i32, i32* %inc.spill.addr, align 4
    %inc = add i32 %n.val, 1
    store i32 %inc, i32* %inc.spill.addr, align 4
    tail call void @print(i32 %inc)
    ret void
  }

Whereas function `f.destroy` will contain the cleanup code for the coroutine:

.. code-block:: llvm

  define internal fastcc void @f.destroy(%f.frame* %frame.ptr.destroy) {
  entry:
    %0 = bitcast %f.frame* %frame.ptr.destroy to i8*
    tail call void @free(i8* %0)
    ret void
  }

Avoiding Heap Allocations
-------------------------
 
A particular coroutine usage pattern, which is illustrated by the `main` 
function in the overview section, where a coroutine is created, manipulated and 
destroyed by the same calling function, is common for coroutines implementing
RAII idiom and is suitable for allocation elision optimization which avoid 
dynamic allocation by storing the coroutine frame as a static `alloca` in its 
caller.

In the entry block, we will call `coro.alloc`_ intrinsic that will return `true`
when dynamic allocation is required, and `false` if dynamic allocation is 
elided.

.. code-block:: llvm

  entry:
    %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
    %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
    br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
  dyn.alloc:
    %size = call i32 @llvm.coro.size.i32()
    %alloc = call i8* @CustomAlloc(i32 %size)
    br label %coro.begin
  coro.begin:
    %phi = phi i8* [ null, %entry ], [ %alloc, %dyn.alloc ]
    %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %phi)

In the cleanup block, we will make freeing the coroutine frame conditional on
`coro.free`_ intrinsic. If allocation is elided, `coro.free`_ returns `null`
thus skipping the deallocation code:

.. code-block:: llvm

  cleanup:
    %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
    %need.dyn.free = icmp ne i8* %mem, null
    br i1 %need.dyn.free, label %dyn.free, label %if.end
  dyn.free:
    call void @CustomFree(i8* %mem)
    br label %if.end
  if.end:
    ...

With allocations and deallocations represented as described as above, after
coroutine heap allocation elision optimization, the resulting main will be:

.. code-block:: llvm

  define i32 @main() {
  entry:
    call void @print(i32 4)
    call void @print(i32 5)
    call void @print(i32 6)
    ret i32 0
  }

Multiple Suspend Points
-----------------------

Let's consider the coroutine that has more than one suspend point:

.. code-block:: c++

  void *f(int n) {
     for(;;) {
       print(n++);
       <suspend>
       print(-n);
       <suspend>
     }
  }

Matching LLVM code would look like (with the rest of the code remaining the same
as the code in the previous section):

.. code-block:: llvm

  loop:
    %n.addr = phi i32 [ %n, %entry ], [ %inc, %loop.resume ]
    call void @print(i32 %n.addr) #4
    %2 = call i8 @llvm.coro.suspend(token none, i1 false)
    switch i8 %2, label %suspend [i8 0, label %loop.resume
                                  i8 1, label %cleanup]
  loop.resume:
    %inc = add nsw i32 %n.addr, 1
    %sub = xor i32 %n.addr, -1
    call void @print(i32 %sub)
    %3 = call i8 @llvm.coro.suspend(token none, i1 false)
    switch i8 %3, label %suspend [i8 0, label %loop
                                  i8 1, label %cleanup]

In this case, the coroutine frame would include a suspend index that will 
indicate at which suspend point the coroutine needs to resume. The resume 
function will use an index to jump to an appropriate basic block and will look 
as follows:

.. code-block:: llvm

  define internal fastcc void @f.Resume(%f.Frame* %FramePtr) {
  entry.Resume:
    %index.addr = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i64 0, i32 2
    %index = load i8, i8* %index.addr, align 1
    %switch = icmp eq i8 %index, 0
    %n.addr = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i64 0, i32 3
    %n = load i32, i32* %n.addr, align 4
    br i1 %switch, label %loop.resume, label %loop

  loop.resume:
    %sub = xor i32 %n, -1
    call void @print(i32 %sub)
    br label %suspend
  loop:
    %inc = add nsw i32 %n, 1
    store i32 %inc, i32* %n.addr, align 4
    tail call void @print(i32 %inc)
    br label %suspend

  suspend:
    %storemerge = phi i8 [ 0, %loop ], [ 1, %loop.resume ]
    store i8 %storemerge, i8* %index.addr, align 1
    ret void
  }

If different cleanup code needs to get executed for different suspend points, 
a similar switch will be in the `f.destroy` function.

.. note ::

  Using suspend index in a coroutine state and having a switch in `f.resume` and
  `f.destroy` is one of the possible implementation strategies. We explored 
  another option where a distinct `f.resume1`, `f.resume2`, etc. are created for
  every suspend point, and instead of storing an index, the resume and destroy 
  function pointers are updated at every suspend. Early testing showed that the
  current approach is easier on the optimizer than the latter so it is a 
  lowering strategy implemented at the moment.

Distinct Save and Suspend
-------------------------

In the previous example, setting a resume index (or some other state change that 
needs to happen to prepare a coroutine for resumption) happens at the same time as
a suspension of a coroutine. However, in certain cases, it is necessary to control 
when coroutine is prepared for resumption and when it is suspended.

In the following example, a coroutine represents some activity that is driven
by completions of asynchronous operations `async_op1` and `async_op2` which get
a coroutine handle as a parameter and resume the coroutine once async
operation is finished.

.. code-block:: text

  void g() {
     for (;;)
       if (cond()) {
          async_op1(<coroutine-handle>); // will resume once async_op1 completes
          <suspend>
          do_one();
       }
       else {
          async_op2(<coroutine-handle>); // will resume once async_op2 completes
          <suspend>
          do_two();
       }
     }
  }

In this case, coroutine should be ready for resumption prior to a call to 
`async_op1` and `async_op2`. The `coro.save`_ intrinsic is used to indicate a
point when coroutine should be ready for resumption (namely, when a resume index
should be stored in the coroutine frame, so that it can be resumed at the 
correct resume point):

.. code-block:: llvm

  if.true:
    %save1 = call token @llvm.coro.save(i8* %hdl)
    call void @async_op1(i8* %hdl)
    %suspend1 = call i1 @llvm.coro.suspend(token %save1, i1 false)
    switch i8 %suspend1, label %suspend [i8 0, label %resume1
                                         i8 1, label %cleanup]
  if.false:
    %save2 = call token @llvm.coro.save(i8* %hdl)
    call void @async_op2(i8* %hdl)
    %suspend2 = call i1 @llvm.coro.suspend(token %save2, i1 false)
    switch i8 %suspend1, label %suspend [i8 0, label %resume2
                                         i8 1, label %cleanup]

.. _coroutine promise:

Coroutine Promise
-----------------

A coroutine author or a frontend may designate a distinguished `alloca` that can
be used to communicate with the coroutine. This distinguished alloca is called
**coroutine promise** and is provided as the second parameter to the 
`coro.id`_ intrinsic.

The following coroutine designates a 32 bit integer `promise` and uses it to
store the current value produced by a coroutine.

.. code-block:: llvm

  define i8* @f(i32 %n) {
  entry:
    %promise = alloca i32
    %pv = bitcast i32* %promise to i8*
    %id = call token @llvm.coro.id(i32 0, i8* %pv, i8* null, i8* null)
    %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
    br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
  dyn.alloc:
    %size = call i32 @llvm.coro.size.i32()
    %alloc = call i8* @malloc(i32 %size)
    br label %coro.begin
  coro.begin:
    %phi = phi i8* [ null, %entry ], [ %alloc, %dyn.alloc ]
    %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %phi)
    br label %loop
  loop:
    %n.val = phi i32 [ %n, %coro.begin ], [ %inc, %loop ]
    %inc = add nsw i32 %n.val, 1
    store i32 %n.val, i32* %promise
    %0 = call i8 @llvm.coro.suspend(token none, i1 false)
    switch i8 %0, label %suspend [i8 0, label %loop
                                  i8 1, label %cleanup]
  cleanup:
    %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
    call void @free(i8* %mem)
    br label %suspend
  suspend:
    %unused = call i1 @llvm.coro.end(i8* %hdl, i1 false)
    ret i8* %hdl
  }

A coroutine consumer can rely on the `coro.promise`_ intrinsic to access the
coroutine promise.

.. code-block:: llvm

  define i32 @main() {
  entry:
    %hdl = call i8* @f(i32 4)
    %promise.addr.raw = call i8* @llvm.coro.promise(i8* %hdl, i32 4, i1 false)
    %promise.addr = bitcast i8* %promise.addr.raw to i32*
    %val0 = load i32, i32* %promise.addr
    call void @print(i32 %val0)
    call void @llvm.coro.resume(i8* %hdl)
    %val1 = load i32, i32* %promise.addr
    call void @print(i32 %val1)
    call void @llvm.coro.resume(i8* %hdl)
    %val2 = load i32, i32* %promise.addr
    call void @print(i32 %val2)
    call void @llvm.coro.destroy(i8* %hdl)
    ret i32 0
  }

After example in this section is compiled, result of the compilation will be:

.. code-block:: llvm

  define i32 @main() {
  entry:
    tail call void @print(i32 4)
    tail call void @print(i32 5)
    tail call void @print(i32 6)
    ret i32 0
  }

.. _final:
.. _final suspend:

Final Suspend
-------------

A coroutine author or a frontend may designate a particular suspend to be final,
by setting the second argument of the `coro.suspend`_ intrinsic to `true`.
Such a suspend point has two properties:

* it is possible to check whether a suspended coroutine is at the final suspend
  point via `coro.done`_ intrinsic;

* a resumption of a coroutine stopped at the final suspend point leads to 
  undefined behavior. The only possible action for a coroutine at a final
  suspend point is destroying it via `coro.destroy`_ intrinsic.

From the user perspective, the final suspend point represents an idea of a 
coroutine reaching the end. From the compiler perspective, it is an optimization
opportunity for reducing number of resume points (and therefore switch cases) in
the resume function.

The following is an example of a function that keeps resuming the coroutine
until the final suspend point is reached after which point the coroutine is 
destroyed:

.. code-block:: llvm

  define i32 @main() {
  entry:
    %hdl = call i8* @f(i32 4)
    br label %while
  while:
    call void @llvm.coro.resume(i8* %hdl)
    %done = call i1 @llvm.coro.done(i8* %hdl)
    br i1 %done, label %end, label %while
  end:
    call void @llvm.coro.destroy(i8* %hdl)
    ret i32 0
  }

Usually, final suspend point is a frontend injected suspend point that does not
correspond to any explicitly authored suspend point of the high level language.
For example, for a Python generator that has only one suspend point:

.. code-block:: python

  def coroutine(n):
    for i in range(n):
      yield i

Python frontend would inject two more suspend points, so that the actual code
looks like this:

.. code-block:: c

  void* coroutine(int n) {
    int current_value; 
    <designate current_value to be coroutine promise>
    <SUSPEND> // injected suspend point, so that the coroutine starts suspended
    for (int i = 0; i < n; ++i) {
      current_value = i; <SUSPEND>; // corresponds to "yield i"
    }
    <SUSPEND final=true> // injected final suspend point
  }

and python iterator `__next__` would look like:

.. code-block:: c++

  int __next__(void* hdl) {
    coro.resume(hdl);
    if (coro.done(hdl)) throw StopIteration();
    return *(int*)coro.promise(hdl, 4, false);
  }


Intrinsics
==========

Coroutine Manipulation Intrinsics
---------------------------------

Intrinsics described in this section are used to manipulate an existing
coroutine. They can be used in any function which happen to have a pointer
to a `coroutine frame`_ or a pointer to a `coroutine promise`_.

.. _coro.destroy:

'llvm.coro.destroy' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

::

      declare void @llvm.coro.destroy(i8* <handle>)

Overview:
"""""""""

The '``llvm.coro.destroy``' intrinsic destroys a suspended
switched-resume coroutine.

Arguments:
""""""""""

The argument is a coroutine handle to a suspended coroutine.

Semantics:
""""""""""

When possible, the `coro.destroy` intrinsic is replaced with a direct call to 
the coroutine destroy function. Otherwise it is replaced with an indirect call 
based on the function pointer for the destroy function stored in the coroutine
frame. Destroying a coroutine that is not suspended leads to undefined behavior.

.. _coro.resume:

'llvm.coro.resume' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

      declare void @llvm.coro.resume(i8* <handle>)

Overview:
"""""""""

The '``llvm.coro.resume``' intrinsic resumes a suspended switched-resume coroutine.

Arguments:
""""""""""

The argument is a handle to a suspended coroutine.

Semantics:
""""""""""

When possible, the `coro.resume` intrinsic is replaced with a direct call to the
coroutine resume function. Otherwise it is replaced with an indirect call based 
on the function pointer for the resume function stored in the coroutine frame. 
Resuming a coroutine that is not suspended leads to undefined behavior.

.. _coro.done:

'llvm.coro.done' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

      declare i1 @llvm.coro.done(i8* <handle>)

Overview:
"""""""""

The '``llvm.coro.done``' intrinsic checks whether a suspended
switched-resume coroutine is at the final suspend point or not.

Arguments:
""""""""""

The argument is a handle to a suspended coroutine.

Semantics:
""""""""""

Using this intrinsic on a coroutine that does not have a `final suspend`_ point 
or on a coroutine that is not suspended leads to undefined behavior.

.. _coro.promise:

'llvm.coro.promise' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

      declare i8* @llvm.coro.promise(i8* <ptr>, i32 <alignment>, i1 <from>)

Overview:
"""""""""

The '``llvm.coro.promise``' intrinsic obtains a pointer to a 
`coroutine promise`_ given a switched-resume coroutine handle and vice versa.

Arguments:
""""""""""

The first argument is a handle to a coroutine if `from` is false. Otherwise, 
it is a pointer to a coroutine promise.

The second argument is an alignment requirements of the promise. 
If a frontend designated `%promise = alloca i32` as a promise, the alignment 
argument to `coro.promise` should be the alignment of `i32` on the target 
platform. If a frontend designated `%promise = alloca i32, align 16` as a 
promise, the alignment argument should be 16.
This argument only accepts constants.

The third argument is a boolean indicating a direction of the transformation.
If `from` is true, the intrinsic returns a coroutine handle given a pointer 
to a promise. If `from` is false, the intrinsics return a pointer to a promise 
from a coroutine handle. This argument only accepts constants.

Semantics:
""""""""""

Using this intrinsic on a coroutine that does not have a coroutine promise
leads to undefined behavior. It is possible to read and modify coroutine
promise of the coroutine which is currently executing. The coroutine author and
a coroutine user are responsible to makes sure there is no data races.

Example:
""""""""

.. code-block:: llvm

  define i8* @f(i32 %n) {
  entry:
    %promise = alloca i32
    %pv = bitcast i32* %promise to i8*
    ; the second argument to coro.id points to the coroutine promise.
    %id = call token @llvm.coro.id(i32 0, i8* %pv, i8* null, i8* null)
    ...
    %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %alloc)
    ...
    store i32 42, i32* %promise ; store something into the promise
    ...
    ret i8* %hdl
  }

  define i32 @main() {
  entry:
    %hdl = call i8* @f(i32 4) ; starts the coroutine and returns its handle
    %promise.addr.raw = call i8* @llvm.coro.promise(i8* %hdl, i32 4, i1 false)
    %promise.addr = bitcast i8* %promise.addr.raw to i32*    
    %val = load i32, i32* %promise.addr ; load a value from the promise
    call void @print(i32 %val)
    call void @llvm.coro.destroy(i8* %hdl)
    ret i32 0
  }

.. _coroutine intrinsics:

Coroutine Structure Intrinsics
------------------------------
Intrinsics described in this section are used within a coroutine to describe
the coroutine structure. They should not be used outside of a coroutine.

.. _coro.size:

'llvm.coro.size' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

    declare i32 @llvm.coro.size.i32()
    declare i64 @llvm.coro.size.i64()

Overview:
"""""""""

The '``llvm.coro.size``' intrinsic returns the number of bytes
required to store a `coroutine frame`_.  This is only supported for
switched-resume coroutines.

Arguments:
""""""""""

None

Semantics:
""""""""""

The `coro.size` intrinsic is lowered to a constant representing the size of
the coroutine frame. 

.. _coro.begin:

'llvm.coro.begin' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i8* @llvm.coro.begin(token <id>, i8* <mem>)

Overview:
"""""""""

The '``llvm.coro.begin``' intrinsic returns an address of the coroutine frame.

Arguments:
""""""""""

The first argument is a token returned by a call to '``llvm.coro.id``' 
identifying the coroutine.

The second argument is a pointer to a block of memory where coroutine frame
will be stored if it is allocated dynamically.  This pointer is ignored
for returned-continuation coroutines.

Semantics:
""""""""""

Depending on the alignment requirements of the objects in the coroutine frame
and/or on the codegen compactness reasons the pointer returned from `coro.begin` 
may be at offset to the `%mem` argument. (This could be beneficial if 
instructions that express relative access to data can be more compactly encoded 
with small positive and negative offsets).

A frontend should emit exactly one `coro.begin` intrinsic per coroutine.

.. _coro.free:

'llvm.coro.free' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i8* @llvm.coro.free(token %id, i8* <frame>)

Overview:
"""""""""

The '``llvm.coro.free``' intrinsic returns a pointer to a block of memory where 
coroutine frame is stored or `null` if this instance of a coroutine did not use
dynamically allocated memory for its coroutine frame.  This intrinsic is not
supported for returned-continuation coroutines.

Arguments:
""""""""""

The first argument is a token returned by a call to '``llvm.coro.id``' 
identifying the coroutine.

The second argument is a pointer to the coroutine frame. This should be the same
pointer that was returned by prior `coro.begin` call.

Example (custom deallocation function):
"""""""""""""""""""""""""""""""""""""""

.. code-block:: llvm

  cleanup:
    %mem = call i8* @llvm.coro.free(token %id, i8* %frame)
    %mem_not_null = icmp ne i8* %mem, null
    br i1 %mem_not_null, label %if.then, label %if.end
  if.then:
    call void @CustomFree(i8* %mem)
    br label %if.end
  if.end:
    ret void

Example (standard deallocation functions):
""""""""""""""""""""""""""""""""""""""""""

.. code-block:: llvm

  cleanup:
    %mem = call i8* @llvm.coro.free(token %id, i8* %frame)
    call void @free(i8* %mem)
    ret void

.. _coro.alloc:

'llvm.coro.alloc' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i1 @llvm.coro.alloc(token <id>)

Overview:
"""""""""

The '``llvm.coro.alloc``' intrinsic returns `true` if dynamic allocation is
required to obtain a memory for the coroutine frame and `false` otherwise.
This is not supported for returned-continuation coroutines.

Arguments:
""""""""""

The first argument is a token returned by a call to '``llvm.coro.id``' 
identifying the coroutine.

Semantics:
""""""""""

A frontend should emit at most one `coro.alloc` intrinsic per coroutine.
The intrinsic is used to suppress dynamic allocation of the coroutine frame
when possible.

Example:
""""""""

.. code-block:: llvm

  entry:
    %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
    %dyn.alloc.required = call i1 @llvm.coro.alloc(token %id)
    br i1 %dyn.alloc.required, label %coro.alloc, label %coro.begin

  coro.alloc:
    %frame.size = call i32 @llvm.coro.size()
    %alloc = call i8* @MyAlloc(i32 %frame.size)
    br label %coro.begin

  coro.begin:
    %phi = phi i8* [ null, %entry ], [ %alloc, %coro.alloc ]
    %frame = call i8* @llvm.coro.begin(token %id, i8* %phi)

.. _coro.noop:

'llvm.coro.noop' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i8* @llvm.coro.noop()

Overview:
"""""""""

The '``llvm.coro.noop``' intrinsic returns an address of the coroutine frame of
a coroutine that does nothing when resumed or destroyed.

Arguments:
""""""""""

None

Semantics:
""""""""""

This intrinsic is lowered to refer to a private constant coroutine frame. The
resume and destroy handlers for this frame are empty functions that do nothing.
Note that in different translation units llvm.coro.noop may return different pointers.

.. _coro.frame:

'llvm.coro.frame' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i8* @llvm.coro.frame()

Overview:
"""""""""

The '``llvm.coro.frame``' intrinsic returns an address of the coroutine frame of
the enclosing coroutine.

Arguments:
""""""""""

None

Semantics:
""""""""""

This intrinsic is lowered to refer to the `coro.begin`_ instruction. This is
a frontend convenience intrinsic that makes it easier to refer to the
coroutine frame.

.. _coro.id:

'llvm.coro.id' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare token @llvm.coro.id(i32 <align>, i8* <promise>, i8* <coroaddr>, 
                                                          i8* <fnaddrs>)

Overview:
"""""""""

The '``llvm.coro.id``' intrinsic returns a token identifying a
switched-resume coroutine.

Arguments:
""""""""""

The first argument provides information on the alignment of the memory returned 
by the allocation function and given to `coro.begin` by the first argument. If 
this argument is 0, the memory is assumed to be aligned to 2 * sizeof(i8*).
This argument only accepts constants.

The second argument, if not `null`, designates a particular alloca instruction
to be a `coroutine promise`_.

The third argument is `null` coming out of the frontend. The CoroEarly pass sets
this argument to point to the function this coro.id belongs to. 

The fourth argument is `null` before coroutine is split, and later is replaced 
to point to a private global constant array containing function pointers to 
outlined resume and destroy parts of the coroutine.


Semantics:
""""""""""

The purpose of this intrinsic is to tie together `coro.id`, `coro.alloc` and
`coro.begin` belonging to the same coroutine to prevent optimization passes from
duplicating any of these instructions unless entire body of the coroutine is
duplicated.

A frontend should emit exactly one `coro.id` intrinsic per coroutine.

.. _coro.id.async:

'llvm.coro.id.async' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare token @llvm.coro.id.async(i32 <context size>, i32 <align>,
                                    i8* <context arg>,
                                    i8* <async function pointer>)

Overview:
"""""""""

The '``llvm.coro.id.async``' intrinsic returns a token identifying an async coroutine.

Arguments:
""""""""""

The first argument provides the initial size of the `async context` as required
from the frontend. Lowering will add to this size the size required by the frame
storage and store that value to the `async function pointer`.

The second argument, is the alignment guarantee of the memory of the
`async context`. The frontend guarantees that the memory will be aligned by this
value.

The third argument is the `async context` argument in the current coroutine.

The fourth argument is the address of the `async function pointer` struct.
Lowering will update the context size requirement in this struct by adding the
coroutine frame size requirement to the initial size requirement as specified by
the first argument of this intrinsic.


Semantics:
""""""""""

A frontend should emit exactly one `coro.id.async` intrinsic per coroutine.

.. _coro.id.retcon:

'llvm.coro.id.retcon' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare token @llvm.coro.id.retcon(i32 <size>, i32 <align>, i8* <buffer>,
                                     i8* <continuation prototype>,
                                     i8* <alloc>, i8* <dealloc>)

Overview:
"""""""""

The '``llvm.coro.id.retcon``' intrinsic returns a token identifying a
multiple-suspend returned-continuation coroutine.

The 'result-type sequence' of the coroutine is defined as follows:

- if the return type of the coroutine function is ``void``, it is the
  empty sequence;

- if the return type of the coroutine function is a ``struct``, it is the
  element types of that ``struct`` in order;

- otherwise, it is just the return type of the coroutine function.

The first element of the result-type sequence must be a pointer type;
continuation functions will be coerced to this type.  The rest of
the sequence are the 'yield types', and any suspends in the coroutine
must take arguments of these types.

Arguments:
""""""""""

The first and second arguments are the expected size and alignment of
the buffer provided as the third argument.  They must be constant.

The fourth argument must be a reference to a global function, called
the 'continuation prototype function'.  The type, calling convention,
and attributes of any continuation functions will be taken from this
declaration.  The return type of the prototype function must match the
return type of the current function.  The first parameter type must be
a pointer type.  The second parameter type must be an integer type;
it will be used only as a boolean flag.

The fifth argument must be a reference to a global function that will
be used to allocate memory.  It may not fail, either by returning null
or throwing an exception.  It must take an integer and return a pointer.

The sixth argument must be a reference to a global function that will
be used to deallocate memory.  It must take a pointer and return ``void``.

'llvm.coro.id.retcon.once' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare token @llvm.coro.id.retcon.once(i32 <size>, i32 <align>, i8* <buffer>,
                                          i8* <prototype>,
                                          i8* <alloc>, i8* <dealloc>)

Overview:
"""""""""

The '``llvm.coro.id.retcon.once``' intrinsic returns a token identifying a
unique-suspend returned-continuation coroutine.

Arguments:
""""""""""

As for ``llvm.core.id.retcon``, except that the return type of the
continuation prototype must be `void` instead of matching the
coroutine's return type.

.. _coro.end:

'llvm.coro.end' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i1 @llvm.coro.end(i8* <handle>, i1 <unwind>)

Overview:
"""""""""

The '``llvm.coro.end``' marks the point where execution of the resume part of 
the coroutine should end and control should return to the caller.


Arguments:
""""""""""

The first argument should refer to the coroutine handle of the enclosing
coroutine. A frontend is allowed to supply null as the first parameter, in this
case `coro-early` pass will replace the null with an appropriate coroutine 
handle value.

The second argument should be `true` if this coro.end is in the block that is 
part of the unwind sequence leaving the coroutine body due to an exception and 
`false` otherwise.

Semantics:
""""""""""
The purpose of this intrinsic is to allow frontends to mark the cleanup and
other code that is only relevant during the initial invocation of the coroutine
and should not be present in resume and destroy parts. 

In returned-continuation lowering, ``llvm.coro.end`` fully destroys the
coroutine frame.  If the second argument is `false`, it also returns from
the coroutine with a null continuation pointer, and the next instruction
will be unreachable.  If the second argument is `true`, it falls through
so that the following logic can resume unwinding.  In a yield-once
coroutine, reaching a non-unwind ``llvm.coro.end`` without having first
reached a ``llvm.coro.suspend.retcon`` has undefined behavior.

The remainder of this section describes the behavior under switched-resume
lowering.

This intrinsic is lowered when a coroutine is split into
the start, resume and destroy parts. In the start part, it is a no-op,
in resume and destroy parts, it is replaced with `ret void` instruction and
the rest of the block containing `coro.end` instruction is discarded.
In landing pads it is replaced with an appropriate instruction to unwind to 
caller. The handling of coro.end differs depending on whether the target is 
using landingpad or WinEH exception model.

For landingpad based exception model, it is expected that frontend uses the 
`coro.end`_ intrinsic as follows:

.. code-block:: llvm

    ehcleanup:
      %InResumePart = call i1 @llvm.coro.end(i8* null, i1 true)
      br i1 %InResumePart, label %eh.resume, label %cleanup.cont

    cleanup.cont:
      ; rest of the cleanup

    eh.resume:
      %exn = load i8*, i8** %exn.slot, align 8
      %sel = load i32, i32* %ehselector.slot, align 4
      %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn, 0
      %lpad.val29 = insertvalue { i8*, i32 } %lpad.val, i32 %sel, 1
      resume { i8*, i32 } %lpad.val29

The `CoroSpit` pass replaces `coro.end` with ``True`` in the resume functions,
thus leading to immediate unwind to the caller, whereas in start function it
is replaced with ``False``, thus allowing to proceed to the rest of the cleanup
code that is only needed during initial invocation of the coroutine.

For Windows Exception handling model, a frontend should attach a funclet bundle
referring to an enclosing cleanuppad as follows:

.. code-block:: llvm

    ehcleanup: 
      %tok = cleanuppad within none []
      %unused = call i1 @llvm.coro.end(i8* null, i1 true) [ "funclet"(token %tok) ]
      cleanupret from %tok unwind label %RestOfTheCleanup

The `CoroSplit` pass, if the funclet bundle is present, will insert 
``cleanupret from %tok unwind to caller`` before
the `coro.end`_ intrinsic and will remove the rest of the block.

The following table summarizes the handling of `coro.end`_ intrinsic.

+--------------------------+-------------------+-------------------------------+
|                          | In Start Function | In Resume/Destroy Functions   |
+--------------------------+-------------------+-------------------------------+
|unwind=false              | nothing           |``ret void``                   |
+------------+-------------+-------------------+-------------------------------+
|            | WinEH       | nothing           |``cleanupret unwind to caller``|
|unwind=true +-------------+-------------------+-------------------------------+
|            | Landingpad  | nothing           | nothing                       |
+------------+-------------+-------------------+-------------------------------+


'llvm.coro.end.async' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i1 @llvm.coro.end.async(i8* <handle>, i1 <unwind>, ...)

Overview:
"""""""""

The '``llvm.coro.end.async``' marks the point where execution of the resume part
of the coroutine should end and control should return to the caller. As part of
its variable tail arguments this instruction allows to specify a function and
the function's arguments that are to be tail called as the last action before
returning.


Arguments:
""""""""""

The first argument should refer to the coroutine handle of the enclosing
coroutine. A frontend is allowed to supply null as the first parameter, in this
case `coro-early` pass will replace the null with an appropriate coroutine
handle value.

The second argument should be `true` if this coro.end is in the block that is
part of the unwind sequence leaving the coroutine body due to an exception and
`false` otherwise.

The third argument if present should specify a function to be called.

If the third argument is present, the remaining arguments are the arguments to
the function call.

.. code-block:: llvm

  call i1 (i8*, i1, ...) @llvm.coro.end.async(
                           i8* %hdl, i1 0,
                           void (i8*, %async.task*, %async.actor*)* @must_tail_call_return,
                           i8* %ctxt, %async.task* %task, %async.actor* %actor)
  unreachable

.. _coro.suspend:
.. _suspend points:

'llvm.coro.suspend' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i8 @llvm.coro.suspend(token <save>, i1 <final>)

Overview:
"""""""""

The '``llvm.coro.suspend``' marks the point where execution of a
switched-resume coroutine is suspended and control is returned back
to the caller.  Conditional branches consuming the result of this
intrinsic lead to basic blocks where coroutine should proceed when
suspended (-1), resumed (0) or destroyed (1).

Arguments:
""""""""""

The first argument refers to a token of `coro.save` intrinsic that marks the 
point when coroutine state is prepared for suspension. If `none` token is passed,
the intrinsic behaves as if there were a `coro.save` immediately preceding
the `coro.suspend` intrinsic.

The second argument indicates whether this suspension point is `final`_.
The second argument only accepts constants. If more than one suspend point is
designated as final, the resume and destroy branches should lead to the same
basic blocks.

Example (normal suspend point):
"""""""""""""""""""""""""""""""

.. code-block:: llvm

    %0 = call i8 @llvm.coro.suspend(token none, i1 false)
    switch i8 %0, label %suspend [i8 0, label %resume
                                  i8 1, label %cleanup]

Example (final suspend point):
""""""""""""""""""""""""""""""

.. code-block:: llvm

  while.end:
    %s.final = call i8 @llvm.coro.suspend(token none, i1 true)
    switch i8 %s.final, label %suspend [i8 0, label %trap
                                        i8 1, label %cleanup]
  trap: 
    call void @llvm.trap()
    unreachable

Semantics:
""""""""""

If a coroutine that was suspended at the suspend point marked by this intrinsic
is resumed via `coro.resume`_ the control will transfer to the basic block
of the 0-case. If it is resumed via `coro.destroy`_, it will proceed to the
basic block indicated by the 1-case. To suspend, coroutine proceed to the 
default label.

If suspend intrinsic is marked as final, it can consider the `true` branch
unreachable and can perform optimizations that can take advantage of that fact.

.. _coro.save:

'llvm.coro.save' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare token @llvm.coro.save(i8* <handle>)

Overview:
"""""""""

The '``llvm.coro.save``' marks the point where a coroutine need to update its 
state to prepare for resumption to be considered suspended (and thus eligible 
for resumption). 

Arguments:
""""""""""

The first argument points to a coroutine handle of the enclosing coroutine.

Semantics:
""""""""""

Whatever coroutine state changes are required to enable resumption of
the coroutine from the corresponding suspend point should be done at the point 
of `coro.save` intrinsic.

Example:
""""""""

Separate save and suspend points are necessary when a coroutine is used to 
represent an asynchronous control flow driven by callbacks representing
completions of asynchronous operations.

In such a case, a coroutine should be ready for resumption prior to a call to 
`async_op` function that may trigger resumption of a coroutine from the same or
a different thread possibly prior to `async_op` call returning control back
to the coroutine:

.. code-block:: llvm

    %save1 = call token @llvm.coro.save(i8* %hdl)
    call void @async_op1(i8* %hdl)
    %suspend1 = call i1 @llvm.coro.suspend(token %save1, i1 false)
    switch i8 %suspend1, label %suspend [i8 0, label %resume1
                                         i8 1, label %cleanup]

.. _coro.suspend.async:

'llvm.coro.suspend.async' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare {i8*, i8*, i8*} @llvm.coro.suspend.async(
                             i8* <resume function>,
                             i8* <context projection function>,
                             ... <function to call>
                             ... <arguments to function>)

Overview:
"""""""""

The '``llvm.coro.suspend.async``' intrinsic marks the point where
execution of a async coroutine is suspended and control is passed to a callee.

Arguments:
""""""""""

The first argument should be the result of the `llvm.coro.async.resume` intrinsic.
Lowering will replace this intrinsic with the resume function for this suspend
point.

The second argument is the `context projection function`. It should describe
how-to restore the `async context` in the continuation function from the first
argument of the continuation function. Its type is `i8* (i8*)`.

The third argument is the function that models transfer to the callee at the
suspend point. It should take 3 arguments. Lowering will `musttail` call this
function.

The fourth to six argument are the arguments for the third argument.

Semantics:
""""""""""

The result of the intrinsic are mapped to the arguments of the resume function.
Execution is suspended at this intrinsic and resumed when the resume function is
called.

.. _coro.prepare.async:

'llvm.coro.prepare.async' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i8* @llvm.coro.prepare.async(i8* <coroutine function>)

Overview:
"""""""""

The '``llvm.coro.prepare.async``' intrinsic is used to block inlining of the
async coroutine until after coroutine splitting.

Arguments:
""""""""""

The first argument should be an async coroutine of type `void (i8*, i8*, i8*)`.
Lowering will replace this intrinsic with its coroutine function argument.

.. _coro.suspend.retcon:

'llvm.coro.suspend.retcon' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i1 @llvm.coro.suspend.retcon(...)

Overview:
"""""""""

The '``llvm.coro.suspend.retcon``' intrinsic marks the point where
execution of a returned-continuation coroutine is suspended and control
is returned back to the caller.

`llvm.coro.suspend.retcon`` does not support separate save points;
they are not useful when the continuation function is not locally
accessible.  That would be a more appropriate feature for a ``passcon``
lowering that is not yet implemented.

Arguments:
""""""""""

The types of the arguments must exactly match the yielded-types sequence
of the coroutine.  They will be turned into return values from the ramp
and continuation functions, along with the next continuation function.

Semantics:
""""""""""

The result of the intrinsic indicates whether the coroutine should resume
abnormally (non-zero).

In a normal coroutine, it is undefined behavior if the coroutine executes
a call to ``llvm.coro.suspend.retcon`` after resuming abnormally.

In a yield-once coroutine, it is undefined behavior if the coroutine
executes a call to ``llvm.coro.suspend.retcon`` after resuming in any way.

.. _coro.param:

'llvm.coro.param' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i1 @llvm.coro.param(i8* <original>, i8* <copy>)

Overview:
"""""""""

The '``llvm.coro.param``' is used by a frontend to mark up the code used to
construct and destruct copies of the parameters. If the optimizer discovers that
a particular parameter copy is not used after any suspends, it can remove the
construction and destruction of the copy by replacing corresponding coro.param
with `i1 false` and replacing any use of the `copy` with the `original`.

Arguments:
""""""""""

The first argument points to an `alloca` storing the value of a parameter to a 
coroutine. 

The second argument points to an `alloca` storing the value of the copy of that
parameter.

Semantics:
""""""""""

The optimizer is free to always replace this intrinsic with `i1 true`.

The optimizer is also allowed to replace it with `i1 false` provided that the 
parameter copy is only used prior to control flow reaching any of the suspend
points. The code that would be DCE'd if the `coro.param` is replaced with 
`i1 false` is not considered to be a use of the parameter copy.

The frontend can emit this intrinsic if its language rules allow for this 
optimization.

Example:
""""""""
Consider the following example. A coroutine takes two parameters `a` and `b`
that has a destructor and a move constructor.

.. code-block:: c++

  struct A { ~A(); A(A&&); bool foo(); void bar(); };

  task<int> f(A a, A b) {
    if (a.foo())
      return 42;

    a.bar();
    co_await read_async(); // introduces suspend point
    b.bar();
  }

Note that, uses of `b` is used after a suspend point and thus must be copied
into a coroutine frame, whereas `a` does not have to, since it never used 
after suspend.

A frontend can create parameter copies for `a` and `b` as follows:

.. code-block:: text

  task<int> f(A a', A b') {
    a = alloca A;
    b = alloca A;
    // move parameters to its copies
    if (coro.param(a', a)) A::A(a, A&& a');
    if (coro.param(b', b)) A::A(b, A&& b');
    ...
    // destroy parameters copies
    if (coro.param(a', a)) A::~A(a);
    if (coro.param(b', b)) A::~A(b);
  }

The optimizer can replace coro.param(a',a) with `i1 false` and replace all uses
of `a` with `a'`, since it is not used after suspend.

The optimizer must replace coro.param(b', b) with `i1 true`, since `b` is used
after suspend and therefore, it has to reside in the coroutine frame.

Coroutine Transformation Passes
===============================
CoroEarly
---------
The pass CoroEarly lowers coroutine intrinsics that hide the details of the
structure of the coroutine frame, but, otherwise not needed to be preserved to
help later coroutine passes. This pass lowers `coro.frame`_, `coro.done`_, 
and `coro.promise`_ intrinsics.

.. _CoroSplit:

CoroSplit
---------
The pass CoroSplit buides coroutine frame and outlines resume and destroy parts 
into separate functions.

CoroElide
---------
The pass CoroElide examines if the inlined coroutine is eligible for heap 
allocation elision optimization. If so, it replaces 
`coro.begin` intrinsic with an address of a coroutine frame placed on its caller
and replaces `coro.alloc` and `coro.free` intrinsics with `false` and `null`
respectively to remove the deallocation code. 
This pass also replaces `coro.resume` and `coro.destroy` intrinsics with direct 
calls to resume and destroy functions for a particular coroutine where possible.

CoroCleanup
-----------
This pass runs late to lower all coroutine related intrinsics not replaced by
earlier passes.

Areas Requiring Attention
=========================
#. When coro.suspend returns -1, the coroutine is suspended, and it's possible
   that the coroutine has already been destroyed (hence the frame has been freed).
   We cannot access anything on the frame on the suspend path.
   However there is nothing that prevents the compiler from moving instructions
   along that path (e.g. LICM), which can lead to use-after-free. At the moment
   we disabled LICM for loops that have coro.suspend, but the general problem still
   exists and requires a general solution.

#. Take advantage of the lifetime intrinsics for the data that goes into the
   coroutine frame. Leave lifetime intrinsics as is for the data that stays in
   allocas.

#. The CoroElide optimization pass relies on coroutine ramp function to be
   inlined. It would be beneficial to split the ramp function further to 
   increase the chance that it will get inlined into its caller.

#. Design a convention that would make it possible to apply coroutine heap
   elision optimization across ABI boundaries.

#. Cannot handle coroutines with `inalloca` parameters (used in x86 on Windows).

#. Alignment is ignored by coro.begin and coro.free intrinsics.

#. Make required changes to make sure that coroutine optimizations work with
   LTO.

#. More tests, more tests, more tests
