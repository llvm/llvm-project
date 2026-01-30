# ORC Runtime Design

The ORC runtime provides APIs for *executor* processes in an ORC JIT session
(as opposed to the LLVM ORC libraries which provide APIs for *controller*
processes). This includes support for both JIT'd code itself, and for users
of JIT'd code.

## Background

LLVM's On Request Compilation (ORC) APIs support cross-process loading of JIT'd
code. We call the process that defines and links the JIT'd code the *controller*
and the process that executes JIT'd code the *executor*. Controller processes
will link LLVM's ORC library, and construct a JIT'd program using an
llvm::orc::ExecutionSession instance (typically through an convenience wrapper
like llvm::orc::LLJIT). Executor processes construct an `orc_rt::Session`
object to manage resources for, and access to, JIT'd code within the executor
process.

## APIs

### Session

The Session object is the root object for a JIT'd program. It owns the
ResourceManager instances that manage resources supporting JIT'd code (e.g.
JIT'd memory, unwind info registrations, dynamic library handles, etc.).

The Session object must be constructed prior to adding any JIT'd code, and must
outlive execution of any JIT'd code.

An executor may have more than one Session object, in which case each Session
object must outlive execution of any JIT'd code added to that specific session.

### ControllerAccess

ControllerAccess objects support bidirectional RPC between JIT'd code in the
executor and the ExecutionSession in the controller.

Calls in both directions are to "wrapper functions" with a fixed signature (a
function that takes a blob of bytes and returns a blob of bytes as its result).
ControllerAccess objects can not generally assume anything about the format of
the bytes being sent (their interpretation is up to the called function). The
RPC is not fully symmetric: Calls from the controller to the executor specify
wrapper function *addresses* (i.e. the controller can invoke any code in the
executor). Calls from the executor to the controller specify *tags*, which are
addresses in the executor processes that are associated with handlers in the
controller. This ensures that the executing process can only call deliberately
exposed entry points in the controller.

ControllerAccess objects may be detached before the session ends, at which point
JIT'd code may continue executing, but will receive no further calls from the
controller and can make no further calls to the controller.

### ResourceManager

`ResourceManager` is an interface for classes that manage resources that support
a JIT'd program, for example memory or loaded dylib handles. It provides two
operations: `detach` and `shutdown`. The `shutdown` operation will be called at
`Session` destruction time. The `detach` operation may be called if the
controller detaches: since this means that no further requests for resource
allocation or release will occur prior to the end of the Session
ResourceManagers may implement this operation to abandon any fine-grained
tracking or pre-reserved resources (e.g. address space).

### ExecutionScope and ExecutionScopeGuard

A Session has an *execution scope* that must be retained while JIT'd code is
executing. Retaining the execution scope prevents the Session from completing
shutdown, ensuring that JIT'd code and its supporting resources remain valid.
The Session's shutdown process will block until the execution scope is fully
released.

`ExecutionScopeGuard` is an RAII class that retains the Session's execution
scope. Clients should create an ExecutionScopeGuard before executing any JIT'd
code and hold it until execution completes.

Example usage:

```cpp
void runJittedCode(Session &S) {
  ExecutionScopeGuard ESG(S);  // Retain execution scope
  callSomeJittedFunction();
  // ESG automatically releases when it goes out of scope
}
```

For dispatch mechanisms that handle calls from the controller, the dispatch
mechanism should hold an ExecutionScopeGuard that covers all dispatched work:

```cpp
class MyDispatcher {
  ExecutionScopeGuard ESG;
  // ... thread pool or other dispatch mechanism ...
public:
  MyDispatcher(Session &S) : ESG(S) {}
  void dispatch(std::function<void()> Task) {
    // Tasks run under the dispatcher's retained execution scope
    threadPool.enqueue(std::move(Task));
  }
};
```

ExecutionScopeGuards are copyable (each copy independently retains the
execution scope) and movable (transfers the retain). The Session waits for the
execution scope retain count to reach zero before proceeding with resource
manager shutdown.

### WrapperFunction

A wrapper function is any function with the following C signature:

```c
void (orc_rt_SessionRef Session, uint64_t CallId,
      orc_rt_WrapperFunctionReturn Return,
      orc_rt_WrapperFunctionBuffer ArgBytes);
```

where `orc_rt_WrapperFunctionReturn` and `orc_rt_WrapperFunctionBuffer` are
defined as:

```c
typedef struct {
  orc_rt_WrapperFunctionBufferDataUnion Data;
  size_t Size;
} orc_rt_WrapperFunctionBuffer;

/**
 * Asynchronous return function for an orc-rt wrapper function.
 */
typedef void (*orc_rt_WrapperFunctionReturn)(
    orc_rt_SessionRef Session, uint64_t CallId,
    orc_rt_WrapperFunctionBuffer ResultBytes);
```

The orc_rt::WrapperFunction class provides APIs for implementing and calling
wrapper functions.

### SPSWrapperFunction

An SPS wrapper function is a wrapper function that uses the
SimplePackedSerialization scheme (see documentation in
orc-rt/include/orc-rt/SimplePackedSerialization.h).

## TODO:

Document...

* C API
* Error handling
* RTTI
* ExecutorAddr / ExecutorAddrRange
* SimpleNativeMemoryMap
* Memory Access (unimplemented)
* Platform classes (unimplemented)
* Other utilities
