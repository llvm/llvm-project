; RUN: llc -march=hexagon -mcpu=hexagonv5  < %s | FileCheck %s
; Generate MemOps for V4 and above.


define void @f(i32* %p) nounwind {
entry:
; CHECK:  memw(r{{[0-9]+}}{{ *}}+{{ *}}#40){{ *}}-={{ *}}#1
  %p.addr = alloca i32*, align 4
  store i32* %p, i32** %p.addr, align 4
  %0 = load i32*, i32** %p.addr, align 4
  %add.ptr = getelementptr inbounds i32, i32* %0, i32 10
  %1 = load i32, i32* %add.ptr, align 4
  %sub = sub nsw i32 %1, 1
  store i32 %sub, i32* %add.ptr, align 4
  ret void
}

define void @g(i32* %p, i32 %i) nounwind {
entry:
; CHECK: memw(r{{[0-9]+}}{{ *}}+{{ *}}#40){{ *}}-={{ *}}#1
  %p.addr = alloca i32*, align 4
  %i.addr = alloca i32, align 4
  store i32* %p, i32** %p.addr, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32*, i32** %p.addr, align 4
  %1 = load i32, i32* %i.addr, align 4
  %add.ptr = getelementptr inbounds i32, i32* %0, i32 %1
  %add.ptr1 = getelementptr inbounds i32, i32* %add.ptr, i32 10
  %2 = load i32, i32* %add.ptr1, align 4
  %sub = sub nsw i32 %2, 1
  store i32 %sub, i32* %add.ptr1, align 4
  ret void
}
