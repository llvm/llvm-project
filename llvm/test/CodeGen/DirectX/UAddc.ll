; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; CHECK: %dx.types.i32c = type { i32, i1 }

define noundef i32 @test_UAddc(i32 noundef %a, i32 noundef %b) {
; CHECK-LABEL: define noundef i32 @test_UAddc(
; CHECK-SAME: i32 noundef [[A:%.*]], i32 noundef [[B:%.*]]) {
; CHECK-NEXT:    [[UAddc:%.*]] = call %dx.types.i32c @dx.op.binaryWithCarryOrBorrow.i32(i32 44, i32 [[A]], i32 [[B]])
; CHECK-NEXT:    [[Carry:%.*]] = extractvalue %dx.types.i32c [[UAddc]], 1
; CHECK-NEXT:    [[Sum:%.*]] = extractvalue %dx.types.i32c [[UAddc]], 0
; CHECK-NEXT:    [[CarryZExt:%.*]] = zext i1 [[Carry]] to i32
; CHECK-NEXT:    [[Result:%.*]] = add i32 [[Sum]], [[CarryZExt]]
; CHECK-NEXT:    ret i32 [[Result]]
; 
  %uaddc = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)
  %carry = extractvalue { i32, i1 } %uaddc, 1
  %sum = extractvalue { i32, i1 } %uaddc, 0
  %carry_zext = zext i1 %carry to i32
  %result = add i32 %sum, %carry_zext
  ret i32 %result
}

declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32)
; CHECK: declare %dx.types.i32c @dx.op.binaryWithCarryOrBorrow.i32(i32, i32, i32)

