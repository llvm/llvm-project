; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; Check that we generate new value jump.

@i = global i32 0, align 4
@j = global i32 10, align 4

define i32 @foo(i32 %a) nounwind {
entry:
; CHECK: if (cmp.eq(r{{[0-9]+}}.new,#0)) jump{{.}}
  %addr1 = alloca i32, align 4
  %addr2 = alloca i32, align 4
  %0 = load i32, i32* @i, align 4
  store i32 %0, i32* %addr1, align 4
  call void @bar(i32 1, i32 2)
  %1 = load i32, i32* @j, align 4
  %tobool = icmp ne i32 %1, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:
  call void @baz(i32 1, i32 2)
  br label %if.end

if.else:
  call void @guy(i32 10, i32 20)
  br label %if.end

if.end:
  ret i32 0
}

declare void @guy(i32, i32)
declare void @bar(i32, i32)
declare void @baz(i32, i32)
