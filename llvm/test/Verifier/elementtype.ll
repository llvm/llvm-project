; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @some_function(i32*)

; CHECK: Attribute 'elementtype(i32)' applied to incompatible type!
define void @type_mismatch1() {
  call i32* @llvm.preserve.array.access.index.p0i32.p0i32(i32* null, i32 elementtype(i32) 0, i32 0)
  ret void
}

; CHECK: Attribute 'elementtype' type does not match parameter!
define void @type_mismatch2() {
  call i32* @llvm.preserve.array.access.index.p0i32.p0i32(i32* elementtype(i64) null, i32 0, i32 0)
  ret void
}

; CHECK: Attribute 'elementtype' can only be applied to intrinsics and inline asm.
define void @not_intrinsic() {
  call void @some_function(i32* elementtype(i32) null)
  ret void
}

; CHECK: Attribute 'elementtype' can only be applied to a callsite.
define void @llvm.not_call(i32* elementtype(i32)) {
  ret void
}

define void @elementtype_required() {
; CHECK: Intrinsic requires elementtype attribute on first argument.
  call i32* @llvm.preserve.array.access.index.p0i32.p0i32(i32* null, i32 0, i32 0)
; CHECK: Intrinsic requires elementtype attribute on first argument.
  call i32* @llvm.preserve.struct.access.index.p0i32.p0i32(i32* null, i32 0, i32 0)
  ret void
}

declare i32* @llvm.preserve.array.access.index.p0i32.p0i32(i32*, i32, i32)
declare i32* @llvm.preserve.struct.access.index.p0i32.p0i32(i32*, i32, i32)
