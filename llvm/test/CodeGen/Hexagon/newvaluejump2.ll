; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; Check that we generate new value jump, both registers, with one 
; of the registers as new.

@Reg = common global i8 0, align 1
define i32 @main() nounwind {
entry:
; CHECK: if (cmp.gt(r{{[0-9]+}}.new, r{{[0-9]+}})) jump:{{[t|nt]}} .LBB{{[0-9]+}}_{{[0-9]+}}
  %Reg2 = alloca i8, align 1
  %0 = load i8* %Reg2, align 1
  %conv0 = zext i8 %0 to i32
  %1 = load i8* @Reg, align 1
  %conv1 = zext i8 %1 to i32
  %tobool = icmp sle i32 %conv0, %conv1
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
