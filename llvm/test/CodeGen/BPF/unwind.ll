; RUN: llc -mtriple=bpfel -mcpu=v4 -filetype=asm -o - %s | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -filetype=obj -o %t %s
; RUN: llvm-readelf -S -s %t | FileCheck --check-prefix=OBJ %s

; A function that can unwind says in r3 how it returns and its caller looks at
; r3 after the call. There are no tables and no calls into an unwinder.
; A frontend asks for that with uwtable, like rustc does with -C panic=unwind.

; OBJ-NOT: .bpf_cleanup
; OBJ-NOT: .gcc_except_table
; OBJ-NOT: _Unwind_
; OBJ-NOT: abort
; OBJ-NOT: personality

declare ptr @bpf_alloc(i64, i64) nounwind
declare void @bpf_free(ptr) nounwind
declare void @btree_insert(ptr, ptr, i64)
declare i32 @_Unwind_RaiseException(ptr)
declare void @_Unwind_Resume(ptr) noreturn
declare void @abort() noreturn nounwind
declare i32 @rust_eh_personality(i32, i64, ptr, ptr)

; invoke, cleanup pad, resume. A call of nounwind function is left alone.
define void @stat_inc(ptr %map, i64 %val) uwtable personality ptr @rust_eh_personality {
entry:
  %node = call ptr @bpf_alloc(i64 256, i64 0)
  invoke void @btree_insert(ptr %map, ptr %node, i64 %val)
          to label %done unwind label %cleanup

done:
  ret void

cleanup:
  %lp = landingpad { ptr, i32 } cleanup
  call void @bpf_free(ptr %node)
  resume { ptr, i32 } %lp
}

; CHECK-LABEL: stat_inc:
; CHECK:         call bpf_alloc
; CHECK-NOT:     if r3
; CHECK:         call btree_insert
; CHECK-NEXT:    if r3 != 0 goto [[PAD:.LBB[0-9_]+]]
; CHECK:         r3 = 0
; CHECK-NEXT:    exit
; CHECK:       [[PAD]]:
; CHECK:         call bpf_free
; CHECK-NEXT:    r0 = 0
; CHECK-NEXT:    r3 = 1
; CHECK-NEXT:    exit

; A plain call that may unwind: nothing to clean up, pass it on.
define i64 @passes_on(ptr %map) uwtable {
entry:
  call void @btree_insert(ptr %map, ptr null, i64 1)
  ret i64 7
}

; CHECK-LABEL: passes_on:
; CHECK:         call btree_insert
; CHECK-NEXT:    if r3 != 0 goto [[PAD:.LBB[0-9_]+]]
; CHECK:         r0 = 7
; CHECK-NEXT:    r3 = 0
; CHECK-NEXT:    exit
; CHECK:       [[PAD]]:
; CHECK-NEXT:    r0 = 0
; CHECK-NEXT:    r3 = 1
; CHECK-NEXT:    exit

; catch_unwind: the pad does not resume. Nothing gets out of this function, but
; that is not known without inference, so it still says r3 = 0.
define i64 @catches(ptr %map) uwtable personality ptr @rust_eh_personality {
entry:
  invoke void @btree_insert(ptr %map, ptr null, i64 1)
          to label %ok unwind label %caught

ok:
  ret i64 0

caught:
  %lp = landingpad { ptr, i32 } catch ptr null
  ret i64 1
}

; CHECK-LABEL: catches:
; CHECK:         call btree_insert
; CHECK-NEXT:    if r3 != 0 goto [[PAD:.LBB[0-9_]+]]
; CHECK:         r0 = 0
; CHECK:         r3 = 0
; CHECK-NEXT:    exit
; CHECK:       [[PAD]]:
; CHECK-NEXT:    r0 = 1
; CHECK-NOT:     r3 = 1

; The same after inference: it is nounwind, so it says nothing in r3 and its
; callers do not look, but it still has to look at r3 of its callee.
define i64 @catches_nounwind(ptr %map) nounwind uwtable personality ptr @rust_eh_personality {
entry:
  invoke void @btree_insert(ptr %map, ptr null, i64 1)
          to label %ok unwind label %caught

ok:
  ret i64 0

caught:
  %lp = landingpad { ptr, i32 } catch ptr null
  ret i64 1
}

; CHECK-LABEL: catches_nounwind:
; CHECK:         call btree_insert
; CHECK-NEXT:    if r3 != 0 goto
; CHECK-NOT:     r3 =
; CHECK:         exit

; An empty filter lets nothing through: one more way to say catch-all. That is
; how rustc guards a function that must not unwind, e.g. an extern "C" one.
define i64 @empty_filter(ptr %map) nounwind uwtable personality ptr @rust_eh_personality {
entry:
  invoke void @btree_insert(ptr %map, ptr null, i64 1)
          to label %ok unwind label %terminate

ok:
  ret i64 0

terminate:
  %lp = landingpad { ptr, i32 } filter [0 x ptr] zeroinitializer
  ret i64 2
}

; CHECK-LABEL: empty_filter:
; CHECK:         call btree_insert
; CHECK-NEXT:    if r3 != 0 goto [[PAD:.LBB[0-9_]+]]
; CHECK:       [[PAD]]:
; CHECK-NEXT:    r0 = 2

; The start of an unwind, as at the end of a panic handler. It is not different
; from carrying one on. By its prototype _Unwind_RaiseException() comes back
; when it fails, so a runtime has something after it. Here it cannot fail: that
; code is dropped and the function ends with exit.
define void @fail() noreturn uwtable {
entry:
  %err = call i32 @_Unwind_RaiseException(ptr null)
  call void @abort()
  unreachable
}

; CHECK-LABEL: fail:
; CHECK-NOT:     call
; CHECK:         r0 = 0
; CHECK-NEXT:    r3 = 1
; CHECK-NEXT:    exit
; CHECK-NOT:     call
; CHECK:       .Lfunc_end

; The same in a frame that owns something: its own pad comes first.
define void @fail_owner(ptr %node) uwtable personality ptr @rust_eh_personality {
entry:
  %err = invoke i32 @_Unwind_RaiseException(ptr null)
          to label %failed unwind label %cleanup

failed:
  call void @abort()
  unreachable

cleanup:
  %lp = landingpad { ptr, i32 } cleanup
  call void @bpf_free(ptr %node)
  resume { ptr, i32 } %lp
}

; CHECK-LABEL: fail_owner:
; CHECK-NOT:     _Unwind_
; CHECK-NOT:     exit
; CHECK:         call bpf_free
; CHECK-NEXT:    r0 = 0
; CHECK-NEXT:    r3 = 1
; CHECK-NEXT:    exit
; CHECK-NOT:     call
; CHECK:       .Lfunc_end

; What is behind it can be a loop, e.g. 'loop {}' in Rust. The verifier rejects
; unreachable code, so it has to go as well.
define void @fail_then_spins() noreturn uwtable {
entry:
  %err = call i32 @_Unwind_RaiseException(ptr null)
  br label %spin

spin:
  br label %spin
}

; CHECK-LABEL: fail_then_spins:
; CHECK:         r0 = 0
; CHECK-NEXT:    r3 = 1
; CHECK-NEXT:    exit
; CHECK-NOT:     goto
; CHECK:       .Lfunc_end

; 'resume' spelled as the call that other targets lower it to.
define void @resume_spelled_out(ptr %exn) uwtable {
entry:
  call void @_Unwind_Resume(ptr %exn)
  unreachable
}

; CHECK-LABEL: resume_spelled_out:
; CHECK-NOT:     call
; CHECK:         r0 = 0
; CHECK-NEXT:    r3 = 1
; CHECK-NEXT:    exit

; A callee that does not return: no need to look at r3, and the function does
; not end with a call.
define i32 @check(i32 %i) uwtable {
entry:
  %c = icmp ugt i32 %i, 7
  br i1 %c, label %bad, label %good

bad:
  call void @fail()
  unreachable

good:
  ret i32 %i
}

; CHECK-LABEL: check:
; CHECK:         call fail
; CHECK-NOT:     if r3
; CHECK:         r3 = 1
; CHECK-NEXT:    exit

; An indirect call looks at r3 whatever the callee is, so a nounwind function
; whose address is taken has to set it.
define internal i32 @plain(i32 %x) nounwind uwtable {
  %r = mul i32 %x, 3
  ret i32 %r
}

@ops = constant [1 x ptr] [ptr @plain]

define i32 @indirect(i64 %i, i32 %x) uwtable {
entry:
  %p = getelementptr [1 x ptr], ptr @ops, i64 0, i64 %i
  %f = load volatile ptr, ptr %p
  %r = call i32 %f(i32 %x)
  ret i32 %r
}

; CHECK-LABEL: plain:
; CHECK:         r3 = 0
; CHECK-NEXT:    exit
; CHECK-LABEL: indirect:
; CHECK:         callx
; CHECK-NEXT:    if r3 != 0 goto

; nounwind, address not taken: nothing changes.
define i32 @leaf(i32 %x) nounwind uwtable {
  %p = call ptr @bpf_alloc(i64 8, i64 0)
  call void @bpf_free(ptr %p)
  ret i32 %x
}

; CHECK-LABEL: leaf:
; CHECK-NOT:     r3
; CHECK:         exit

; Not being nounwind is the default in IR. Without uwtable nothing changes.
define i32 @not_asked(ptr %map) {
  call void @btree_insert(ptr %map, ptr null, i64 1)
  ret i32 0
}

; CHECK-LABEL: not_asked:
; CHECK:         call btree_insert
; CHECK-NEXT:    {{[rw]}}0 = 0
; CHECK-NEXT:    exit
