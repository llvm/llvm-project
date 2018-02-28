; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+lsl-fast | FileCheck %s

%struct.a = type [256 x i16]
%struct.b = type [256 x i32]
%struct.c = type [256 x i64]

declare void @foo()
define i16 @halfword(%struct.a* %ctx, i32 %xor72) nounwind {
; CHECK-LABEL: halfword:
; CHECK: ubfx [[REG:x[0-9]+]], x1, #9, #8
; CHECK: ldrh [[REG1:w[0-9]+]], [{{.*}}[[REG2:x[0-9]+]], [[REG]], lsl #1]
; CHECK: mov [[REG3:x[0-9]+]], [[REG2]]
; CHECK: strh [[REG1]], [{{.*}}[[REG3]], [[REG]], lsl #1]
  %shr81 = lshr i32 %xor72, 9
  %conv82 = zext i32 %shr81 to i64
  %idxprom83 = and i64 %conv82, 255
  %arrayidx86 = getelementptr inbounds %struct.a, %struct.a* %ctx, i64 0, i64 %idxprom83
  %result = load i16, i16* %arrayidx86, align 2
  call void @foo()
  store i16 %result, i16* %arrayidx86, align 2
  ret i16 %result
}

define i32 @word(%struct.b* %ctx, i32 %xor72) nounwind {
; CHECK-LABEL: word:
; CHECK: ubfx [[REG:x[0-9]+]], x1, #9, #8
; CHECK: ldr [[REG1:w[0-9]+]], [{{.*}}[[REG2:x[0-9]+]], [[REG]], lsl #2]
; CHECK: mov [[REG3:x[0-9]+]], [[REG2]]
; CHECK: str [[REG1]], [{{.*}}[[REG3]], [[REG]], lsl #2]
  %shr81 = lshr i32 %xor72, 9
  %conv82 = zext i32 %shr81 to i64
  %idxprom83 = and i64 %conv82, 255
  %arrayidx86 = getelementptr inbounds %struct.b, %struct.b* %ctx, i64 0, i64 %idxprom83
  %result = load i32, i32* %arrayidx86, align 4
  call void @foo()
  store i32 %result, i32* %arrayidx86, align 4
  ret i32 %result
}

define i64 @doubleword(%struct.c* %ctx, i32 %xor72) nounwind {
; CHECK-LABEL: doubleword:
; CHECK: ubfx [[REG:x[0-9]+]], x1, #9, #8
; CHECK: ldr [[REG1:x[0-9]+]], [{{.*}}[[REG2:x[0-9]+]], [[REG]], lsl #3]
; CHECK: mov [[REG3:x[0-9]+]], [[REG2]]
; CHECK: str [[REG1]], [{{.*}}[[REG3]], [[REG]], lsl #3]
  %shr81 = lshr i32 %xor72, 9
  %conv82 = zext i32 %shr81 to i64
  %idxprom83 = and i64 %conv82, 255
  %arrayidx86 = getelementptr inbounds %struct.c, %struct.c* %ctx, i64 0, i64 %idxprom83
  %result = load i64, i64* %arrayidx86, align 8
  call void @foo()
  store i64 %result, i64* %arrayidx86, align 8
  ret i64 %result
}

define i64 @multi_use_non_memory(i64 %a, i64 %b) {
; CHECK-LABEL: multi_use_non_memory:
; CHECK: lsl [[REG1:x[0-9]+]], x0, #3
; CHECK-NOT: cmp [[REG1]], x1, lsl # 3
; CHECK-NEXT: lsl [[REG2:x[0-9]+]], x1, #3
; CHECK-NEXT: cmp [[REG1]], [[REG2]]
entry:
  %mul1 = shl i64 %a, 3
  %mul2 = shl i64 %b, 3
  %cmp = icmp slt i64 %mul1, %mul2
  br i1 %cmp, label %truebb, label %falsebb
truebb:
  tail call void @foo()
  unreachable
falsebb:
  %cmp2 = icmp sgt i64 %mul1, %mul2
  br i1 %cmp2, label %exitbb, label %endbb
exitbb:
 ret i64 %mul1
endbb:
 ret i64 %mul2
}
