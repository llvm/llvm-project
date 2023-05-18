; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @some_function(ptr)

; CHECK: Attribute 'elementtype(i32)' applied to incompatible type!
define void @type_mismatch1() {
  call ptr @llvm.preserve.array.access.index.p0.p0(ptr null, i32 elementtype(i32) 0, i32 0)
  ret void
}

; CHECK: Attribute 'elementtype' can only be applied to intrinsics and inline asm.
define void @not_intrinsic() {
  call void @some_function(ptr elementtype(i32) null)
  ret void
}

; CHECK: Attribute 'elementtype' can only be applied to a callsite.
define void @llvm.not_call(ptr elementtype(i32)) {
  ret void
}

define void @elementtype_required() {
; CHECK: Intrinsic requires elementtype attribute on first argument.
  call ptr @llvm.preserve.array.access.index.p0.p0(ptr null, i32 0, i32 0)
; CHECK: Intrinsic requires elementtype attribute on first argument.
  call ptr @llvm.preserve.struct.access.index.p0.p0(ptr null, i32 0, i32 0)
  ret void
}

declare ptr @llvm.preserve.array.access.index.p0.p0(ptr, i32, i32)
declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32, i32)
