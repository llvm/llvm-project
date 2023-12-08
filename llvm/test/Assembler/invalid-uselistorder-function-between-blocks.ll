; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: expected uselistorder directive

define i32 @f32(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  br label %first

; <label 0>:
  %eh = mul i32 %e, %1
  %sum = add i32 %eh, %ef
  br label %preexit

preexit:
  %product = phi i32 [%ef, %first], [%sum, %0]
  %backto0 = icmp slt i32 %product, -9
  br i1 %backto0, label %0, label %exit

first:
  %e = add i32 %a, 7
  %f = add i32 %b, 7
  %g = add i32 %c, 8
  %1 = add i32 %d, 8
  %ef = mul i32 %e, %f
  %g1 = mul i32 %g, %1
  %goto0 = icmp slt i32 %g1, -9
  br i1 %goto0, label %0, label %preexit

; uselistorder directives
  uselistorder i32 7, { 1, 0 }
  uselistorder i32 %1, { 1, 0 }
  uselistorder i32 %e, { 1, 0 }
  uselistorder label %0, { 1, 0 }
  uselistorder label %preexit, { 1, 0 }

exit:
  ret i32 %product
}
