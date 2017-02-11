; RUN: llc -march=hexagon -mcpu=hexagonv5 -disable-hexagon-misched < %s \
; RUN:    | FileCheck %s
; Check that we generate new value jump, both registers, with one 
; of the registers as new.

@Reg = common global i32 0, align 4
define i32 @main() nounwind {
entry:
; CHECK: if (cmp.gt(r{{[0-9]+}},r{{[0-9]+}}.new)) jump:{{[t|nt]}} .LBB{{[0-9]+}}_{{[0-9]+}}
  %Reg2 = alloca i32, align 4
  %0 = load i32, i32* %Reg2, align 4
  %1 = load i32, i32* @Reg, align 4
  %tobool = icmp sle i32 %0, %1
  br i1 %tobool, label %if.then, label %if.else

if.then:
  call void @bar(i32 1, i32 2)
  br label %if.end

if.else:
  call void @baz(i32 10, i32 20)
  br label %if.end

if.end:
  ret i32 0
}

declare void @bar(i32, i32)
declare void @baz(i32, i32)
