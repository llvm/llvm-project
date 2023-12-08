; RUN: llc -march=hexagon -mcpu=hexagonv5  < %s | FileCheck %s
; Generate MemOps for V4 and above.


define void @f(ptr %p) nounwind {
entry:
; CHECK:  memw(r{{[0-9]+}}+#40) -= #1
  %p.addr = alloca ptr, align 4
  store ptr %p, ptr %p.addr, align 4
  %0 = load ptr, ptr %p.addr, align 4
  %add.ptr = getelementptr inbounds i32, ptr %0, i32 10
  %1 = load i32, ptr %add.ptr, align 4
  %sub = sub nsw i32 %1, 1
  store i32 %sub, ptr %add.ptr, align 4
  ret void
}

define void @g(ptr %p, i32 %i) nounwind {
entry:
; CHECK: memw(r{{[0-9]+}}+#40) -= #1
  %p.addr = alloca ptr, align 4
  %i.addr = alloca i32, align 4
  store ptr %p, ptr %p.addr, align 4
  store i32 %i, ptr %i.addr, align 4
  %0 = load ptr, ptr %p.addr, align 4
  %1 = load i32, ptr %i.addr, align 4
  %add.ptr = getelementptr inbounds i32, ptr %0, i32 %1
  %add.ptr1 = getelementptr inbounds i32, ptr %add.ptr, i32 10
  %2 = load i32, ptr %add.ptr1, align 4
  %sub = sub nsw i32 %2, 1
  store i32 %sub, ptr %add.ptr1, align 4
  ret void
}
