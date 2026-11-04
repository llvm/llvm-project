; RUN: llc -mtriple=bpfel -mcpu=v4 -filetype=asm -o - %s | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -filetype=obj -o %t %s
; RUN: llvm-readelf -S -s %t | FileCheck --check-prefix=OBJ %s

; A function leaves with exit or with unwind_exit, and unwind_goto after the call
; tells them apart. There are no tables and no calls into an unwinder.
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
; CHECK-NOT:     unwind_goto
; CHECK:         call btree_insert
; CHECK-NEXT:    unwind_goto [[PAD:.LBB[0-9_]+]]
; CHECK-NEXT:  # %bb
; CHECK-NEXT:    exit
; CHECK:       [[PAD]]:
; CHECK:         call bpf_free
; CHECK-NEXT:    r0 = 0
; CHECK-NEXT:    unwind_exit

; A plain call that may unwind: nothing to clean up, pass it on.
define i64 @passes_on(ptr %map) uwtable {
entry:
  call void @btree_insert(ptr %map, ptr null, i64 1)
  ret i64 7
}

; CHECK-LABEL: passes_on:
; CHECK:         call btree_insert
; CHECK-NEXT:    unwind_goto [[PAD:.LBB[0-9_]+]]
; CHECK-NEXT:  # %bb
; CHECK-NEXT:    r0 = 7
; CHECK-NEXT:    exit
; CHECK:       [[PAD]]:
; CHECK-NEXT:    r0 = 0
; CHECK-NEXT:    unwind_exit

; catch_unwind: the pad does not resume.
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
; CHECK-NEXT:    unwind_goto [[PAD:.LBB[0-9_]+]]
; CHECK-NEXT:  # %bb
; CHECK-NEXT:    r0 = 0
; CHECK-NEXT:    exit
; CHECK:       [[PAD]]:
; CHECK-NEXT:    r0 = 1
; CHECK-NEXT:    exit

; A pad expects the exception object in r0. There is none.
define void @looks_at_exn(ptr %map) uwtable personality ptr @rust_eh_personality {
entry:
  invoke void @btree_insert(ptr %map, ptr null, i64 1)
          to label %ok unwind label %caught

ok:
  ret void

caught:
  %lp = landingpad { ptr, i32 } catch ptr null
  %exn = extractvalue { ptr, i32 } %lp, 0
  call void @bpf_free(ptr %exn)
  ret void
}

; CHECK-LABEL: looks_at_exn:
; CHECK:         call btree_insert
; CHECK-NEXT:    unwind_goto [[PAD:.LBB[0-9_]+]]
; CHECK:       [[PAD]]:
; CHECK-NEXT:    r0 = 0
; CHECK-NEXT:    r1 = r0
; CHECK-NEXT:    call bpf_free

; The same after inference: it is nounwind, so there is no unwind_goto after the
; calls of it, but it still needs one after the call of its callee.
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
; CHECK-NEXT:    unwind_goto
; CHECK-NOT:     unwind_exit
; CHECK:       .Lfunc_end

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
; CHECK-NEXT:    unwind_goto [[PAD:.LBB[0-9_]+]]
; CHECK:       [[PAD]]:
; CHECK-NEXT:    r0 = 2

; The start of an unwind, as at the end of a panic handler. It is not different
; from carrying one on. By its prototype _Unwind_RaiseException() comes back
; when it fails, so a runtime has something after it. Here it cannot fail: that
; code is dropped and the function ends with unwind_exit.
define void @fail() noreturn uwtable {
entry:
  %err = call i32 @_Unwind_RaiseException(ptr null)
  call void @abort()
  unreachable
}

; CHECK-LABEL: fail:
; CHECK-NOT:     call
; CHECK:         r0 = 0
; CHECK-NEXT:    unwind_exit
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
; CHECK-NEXT:    unwind_exit
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
; CHECK-NEXT:    unwind_exit
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
; CHECK-NEXT:    unwind_exit

; A callee that does not return: its unwind_exit needs unwind_goto all the same.
; What follows is not reached, but the block cannot end there.
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
; CHECK-NEXT:    unwind_goto [[PAD:.LBB[0-9_]+]]
; CHECK-NEXT:    goto [[PAD]]
; CHECK:       [[PAD]]:
; CHECK-NEXT:    r0 = 0
; CHECK-NEXT:    unwind_exit

; An indirect call is followed by unwind_goto whatever the callee is. A nounwind
; callee does not have to know.
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
; CHECK:         w0 = w1
; CHECK-NEXT:    w0 *= 3
; CHECK-NEXT:    exit
; CHECK-LABEL: indirect:
; CHECK:         callx
; CHECK-NEXT:    unwind_goto

; The same from a caller that is nounwind itself, because it catches everything.
define i32 @indirect_catches(i64 %i, i32 %x) nounwind uwtable personality ptr @rust_eh_personality {
entry:
  %p = getelementptr [1 x ptr], ptr @ops, i64 0, i64 %i
  %f = load volatile ptr, ptr %p
  %r = invoke i32 %f(i32 %x)
          to label %ok unwind label %caught

ok:
  ret i32 %r

caught:
  %lp = landingpad { ptr, i32 } catch ptr null
  ret i32 1
}

; CHECK-LABEL: indirect_catches:
; CHECK:         callx
; CHECK-NEXT:    unwind_goto [[PAD:.LBB[0-9_]+]]
; CHECK-NEXT:  # %bb
; CHECK-NEXT:    exit
; CHECK:       [[PAD]]:
; CHECK-NEXT:    w0 = 1
; CHECK-NEXT:    exit

; unwind_exit returns what exit returns. More than 8 bytes are in r0 and r2, and
; the kernel wants both to be set.
define i128 @returns_pair(ptr %map) uwtable {
entry:
  call void @btree_insert(ptr %map, ptr null, i64 1)
  ret i128 7
}

; CHECK-LABEL: returns_pair:
; CHECK:         call btree_insert
; CHECK-NEXT:    unwind_goto [[PAD:.LBB[0-9_]+]]
; CHECK:         exit
; CHECK:       [[PAD]]:
; CHECK-NEXT:    r0 = 0
; CHECK-NEXT:    r2 = 0
; CHECK-NEXT:    unwind_exit

; nounwind: nothing changes.
define i32 @leaf(i32 %x) nounwind uwtable {
  %p = call ptr @bpf_alloc(i64 8, i64 0)
  call void @bpf_free(ptr %p)
  ret i32 %x
}

; CHECK-LABEL: leaf:
; CHECK-NOT:     unwind_
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
