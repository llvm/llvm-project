; RUN: opt < %s -mem2reg -simplifycfg -S | FileCheck -enable-var-scope %s

define i32 @test(i32 %x) {
; CHECK-LABEL: @test
entry:
  %y = alloca i32, align 4
  %tobool = icmp ne i32 %x, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:
; CHECK: if.then:
; CHECK: [[ASM1:%.*]] = call i32 asm "mov $0, #1", "=r"()
  %tmp1 = call i32 asm "mov $0, #1", "=r"() nounwind readnone
  store i32 %tmp1, i32* %y, align 4
  br label %if.end

if.else:
; CHECK: if.else:
; CHECK: [[ASM2:%.*]] = call i32 asm "mov $0, #2", "=r"()
  %tmp2 = call i32 asm "mov $0, #2", "=r"() nounwind readnone
  store i32 %tmp2, i32* %y, align 4
  br label %if.end

if.end:
; CHECK: if.end:
; CHECK: {{%.*}} = phi i32 [ [[ASM1]], %if.then ], [ [[ASM2]], %if.else ]
  %tmp3 = load i32, i32* %y, align 4
  ret i32 %tmp3
}
