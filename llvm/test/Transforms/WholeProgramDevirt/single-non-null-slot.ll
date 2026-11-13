; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility -pass-remarks=wholeprogramdevirt %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

declare { ptr, i1 } @llvm.type.checked.load(ptr, i32, metadata)

; Calling a NULL function pointer is UB, so a NULL slot is not a call target.
; If all but one candidate are NULL, virtual calls should be devirtualized to the 
; one remaining candidate.

; CHECK: remark: <unknown>:0:0: single-impl: devirtualized a call to vfunc1B
; CHECK: remark: <unknown>:0:0: devirtualized vfunc1B
; CHECK-NOT: devirtualized

; This [8 x i8] case mimics a Rust vtable, which uses an array to store a destructor,
; and the slot may be all zeroes if the type does not require a destructor.
@vtableA = internal unnamed_addr constant <{ [8 x i8], ptr }> <{
  [8 x i8] zeroinitializer,
  ptr @vfunc2A
}>, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 8, !"vfunc2.type"}

@vtableB = internal unnamed_addr constant { [2 x ptr] } { [2 x ptr] [
  ptr @vfunc1B,
  ptr @vfunc2B
]}, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 8, !"vfunc2.type"}

; A slot that is neither NULL nor a function still prevents devirtualization.
@vtableC = internal unnamed_addr constant { [1 x ptr] } { [1 x ptr] [
  ptr inttoptr (i64 1 to ptr)
]}, align 8, !type !{i64 0, !"vfunc3.type"}

@vtableD = internal unnamed_addr constant { [1 x ptr] } { [1 x ptr] [
  ptr @vfunc3D
]}, align 8, !type !{i64 0, !"vfunc3.type"}

define internal void @vfunc1B(ptr %this) {
  ret void
}

define internal void @vfunc2A(ptr %this) {
  ret void
}

define internal void @vfunc2B(ptr %this) {
  ret void
}

define internal void @vfunc3D(ptr %this) {
  ret void
}

; CHECK-LABEL: define void @main(
define void @main(ptr %obj) {
  %1 = ptrtoint ptr @vtableA to i64 ; to keep @vtableA alive
  %2 = ptrtoint ptr @vtableB to i64 ; to keep @vtableB alive
  %vtable = load ptr, ptr %obj
  %pair = tail call { ptr, i1 } @llvm.type.checked.load(ptr %vtable, i32 0, metadata !"vfunc1.type")
  %fptr = extractvalue { ptr, i1 } %pair, 0
  ; This call got devirtualized.
  ; CHECK: call void @vfunc1B(ptr %obj)
  call void %fptr(ptr %obj)
  ret void
}

; CHECK-LABEL: define void @not_a_function(
define void @not_a_function(ptr %obj) {
  %1 = ptrtoint ptr @vtableC to i64 ; to keep @vtableC alive
  %2 = ptrtoint ptr @vtableD to i64 ; to keep @vtableD alive
  %vtable = load ptr, ptr %obj
  %pair = tail call { ptr, i1 } @llvm.type.checked.load(ptr %vtable, i32 0, metadata !"vfunc3.type")
  %fptr = extractvalue { ptr, i1 } %pair, 0
  ; This call cannot be devirtualized.
  ; CHECK: call void %{{.*}}(ptr %obj)
  call void %fptr(ptr %obj)
  ret void
}
