; RUN: rm -rf %t
; RUN: mkdir %t
; RUN: cd %t
; RUN: opt %s -passes=dot-dom -disable-output
; RUN: FileCheck %s -input-file=dom.test1.dot -check-prefix=TEST1
; RUN: FileCheck %s -input-file=dom.test2.dot -check-prefix=TEST2

define void @test1() {
; TEST1: digraph "Dominator tree for 'test1' function"
; TEST1-NEXT: label="Dominator tree for 'test1' function"
; TEST1:      Node0x[[EntryID:.*]] [shape=record,label="{entry:
; TEST1-NEXT: Node0x[[EntryID]] -> Node0x[[A_ID:.*]];
; TEST1-NEXT: Node0x[[EntryID]] -> Node0x[[C_ID:.*]];
; TEST1-NEXT: Node0x[[EntryID]] -> Node0x[[B_ID:.*]];
; TEST1-NEXT: Node0x[[A_ID]] [shape=record,label="{a:
; TEST1-NEXT: Node0x[[C_ID]] [shape=record,label="{c:
; TEST1-NEXT: Node0x[[C_ID]] -> Node0x[[D_ID:.*]];
; TEST1-NEXT: Node0x[[C_ID]] -> Node0x[[E_ID:.*]];
; TEST1-NEXT: Node0x[[D_ID]] [shape=record,label="{d:
; TEST1-NEXT: Node0x[[E_ID]] [shape=record,label="{e:
; TEST1-NEXT: Node0x[[B_ID]] [shape=record,label="{b:

entry:
  br i1 undef, label %a, label %b

a:
  br label %c

b:
  br label %c

c:
  br i1 undef, label %d, label %e

d:
  ret void

e:
  ret void
}

define void @test2() {
; TEST2: digraph "Dominator tree for 'test2' function"
; TEST2-NEXT: label="Dominator tree for 'test2' function"
; TEST2: Node0x[[EntryID:.*]] [shape=record,label="{entry:
; TEST2-NEXT: Node0x[[EntryID]] -> Node0x[[A_ID:.*]];
; TEST2-NEXT: Node0x[[A_ID]] [shape=record,label="{a:
; TEST2-NEXT: Node0x[[A_ID]] -> Node0x[[B_ID:.*]];
; TEST2-NEXT: Node0x[[B_ID]] [shape=record,label="{b:
; TEST2-NEXT: Node0x[[B_ID]] -> Node0x[[C_ID:.*]];
; TEST2-NEXT: Node0x[[C_ID]] [shape=record,label="{c:
; TEST2-NEXT: Node0x[[C_ID]] -> Node0x[[D_ID:.*]];
; TEST2-NEXT: Node0x[[C_ID]] -> Node0x[[E_ID:.*]];
; TEST2-NEXT: Node0x[[D_ID]] [shape=record,label="{d:
; TEST2-NEXT: Node0x[[E_ID]] [shape=record,label="{e:

entry:
  br label %a

a:
  br label %b

b:
  br i1 undef, label %a, label %c

c:
  br i1 undef, label %d, label %e

d:
  br i1 undef, label %a, label %e

e:
  ret void
}
