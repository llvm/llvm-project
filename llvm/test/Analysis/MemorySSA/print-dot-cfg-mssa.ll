; RUN: opt -basic-aa -print-memoryssa -dot-cfg-mssa=out.dot -enable-new-pm=0 -analyze < %s 2>&1 > /dev/null
;RUN: FileCheck %s -input-file=out.dot
; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>' -dot-cfg-mssa=out.dot < %s 2>&1 > /dev/null
;RUN: FileCheck %s -input-file=out.dot

; Test -dot-cfg-mssa option for -print-memoryssa.
; Test is based on following C code with some forwarding basic blocks
; added to show that only those blocks with memory ssa comments
; are colourized.

;void g();

;int f(int *p, int *q, int *r) {
;  int i = 0;
;  if (*r)
;    i = 1;
;  else
;    g();
;  *p = *q + 1;
;  if (i)
;    ++i;
;  return *q;
;}

define signext i32 @f(i32* %p, i32* %q, i32* %r) {
entry:
  br label %bb1

bb1:
  %p.addr = alloca i32*, align 8
  %q.addr = alloca i32*, align 8
  %r.addr = alloca i32*, align 8
  %i = alloca i32, align 4
  store i32* %p, i32** %p.addr, align 8
  store i32* %q, i32** %q.addr, align 8
  store i32* %r, i32** %r.addr, align 8
  %0 = bitcast i32* %i to i8*
  store i32 0, i32* %i, align 4
  %1 = load i32*, i32** %r.addr, align 8
  %2 = load i32, i32* %1, align 4
  %tobool = icmp ne i32 %2, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:
  store i32 1, i32* %i, align 4
  br label %bb2

bb2:
  br label %if.end

if.else:
  call void bitcast (void (...)* @g to void ()*)()
  br label %if.end

if.end:
  %3 = load i32*, i32** %q.addr, align 8
  %4 = load i32, i32* %3, align 4
  %add = add nsw i32 %4, 1
  %5 = load i32*, i32** %p.addr, align 8
  store i32 %add, i32* %5, align 4
  %6 = load i32, i32* %i, align 4
  %tobool1 = icmp ne i32 %6, 0
  br i1 %tobool1, label %if.then2, label %if.end3

if.then2:
  %7 = load i32, i32* %i, align 4
  %inc = add nsw i32 %7, 1
  br label %bb3

bb3:
  store i32 %inc, i32* %i, align 4
  br label %if.end3

if.end3:
  br label %bb4

bb4:
  %8 = load i32*, i32** %q.addr, align 8
  %9 = load i32, i32* %8, align 4
  %10 = bitcast i32* %i to i8*
  ret i32 %9
}

declare void @g(...)

; CHECK: digraph "MSSA"
; CHECK-NEXT: label="MSSA";
; CHECK: {{Node0x.* [shape=record,label="{entry:.*}"]}}
; CHECK: {{[shape=record,style=filled, fillcolor=lightpink,label="{bb1:.*1 = MemoryDef(liveOnEntry).*2 = MemoryDef(1).*3 = MemoryDef(2).*4 = MemoryDef(3).*MemoryUse(3).*MemoryUse(liveOnEntry).*}"]}}
; CHECK {{[shape=record,style=filled, fillcolor=lightpink,label="{if.then:.*5 = MemoryDef(4).*}"]}}
; CHECK {{[shape=record,label="{bb2:.*}"]}}
; CHECK {{[shape=record,style=filled, fillcolor=lightpink,label="{if.else:.*6 = MemoryDef(4).*}"]}}
; CHECK {{[shape=record,style=filled, fillcolor=lightpink,label="{if.end:.*10 = MemoryPhi({bb2,5},{if.else,6})/*MemoryUse(2).*MemoryUse(10).*MemoryUse(1).*7 = MemoryDef(10).*MemoryUse(10).*}"]}}
; CHECK {{[shape=record,style=filled, fillcolor=lightpink,label="{if.then2:.*MemoryUse(10).*}"]}}
; CHECK {{[shape=record,style=filled, fillcolor=lightpink,label="{bb3:.*8 = MemoryDef(7).*}"]}}
; CHECK {{[shape=record,style=filled, fillcolor=lightpink,label="{if.end3:.*9 = MemoryPhi({if.end,7},{bb3,8}).*}"]}}
; CHECK {{[shape=record,style=filled, fillcolor=lightpink,label="{bb4:.*MemoryUse(2).*MemoryUse(7).*}"]}}
