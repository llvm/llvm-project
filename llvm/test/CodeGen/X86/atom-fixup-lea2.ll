; RUN: llc < %s -mcpu=atom -mtriple=i686-linux | FileCheck %s

; CHECK:%bb.5
; CHECK-NEXT:leal
; CHECK-NEXT:leal
; CHECK-NEXT:leal
; CHECK-NEXT:movl


; Test for fixup lea pre-emit pass. LEA instructions should be substituted for
; ADD instructions which compute the address and index of the load because they
; precede the load within 5 instructions. An LEA should also be substituted for
; an ADD which computes part of the index because it precedes the index LEA
; within 5 instructions, this substitution is referred to as backwards chaining.

; Original C Code
;struct node_t
;{
;  int k, m, n, p;
;  int * array;
;};

;extern struct node_t getnode();

;int test()
;{
;  int sum = 0;
;  struct node_t n = getnode();
;  if(n.array != 0 && n.p > 0 && n.k > 0 && n.n > 0 && n.m > 0) {
;    sum = ((int*)((int)n.array + n.p) )[ n.k + n.m + n.n ];
;  }
;  return sum;
;}

%struct.node_t = type { i32, i32, i32, i32, ptr }

define i32 @test() {
entry:
  %n = alloca %struct.node_t, align 4
  call void @getnode(ptr sret(%struct.node_t) %n)
  %array = getelementptr inbounds %struct.node_t, ptr %n, i32 0, i32 4
  %0 = load ptr, ptr %array, align 4
  %cmp = icmp eq ptr %0, null
  br i1 %cmp, label %if.end, label %land.lhs.true

land.lhs.true:
  %p = getelementptr inbounds %struct.node_t, ptr %n, i32 0, i32 3
  %1 = load i32, ptr %p, align 4
  %cmp1 = icmp sgt i32 %1, 0
  br i1 %cmp1, label %land.lhs.true2, label %if.end

land.lhs.true2:
  %2 = load i32, ptr %n, align 4
  %cmp3 = icmp sgt i32 %2, 0
  br i1 %cmp3, label %land.lhs.true4, label %if.end

land.lhs.true4:
  %n5 = getelementptr inbounds %struct.node_t, ptr %n, i32 0, i32 2
  %3 = load i32, ptr %n5, align 4
  %cmp6 = icmp sgt i32 %3, 0
  br i1 %cmp6, label %land.lhs.true7, label %if.end

land.lhs.true7:
  %m = getelementptr inbounds %struct.node_t, ptr %n, i32 0, i32 1
  %4 = load i32, ptr %m, align 4
  %cmp8 = icmp sgt i32 %4, 0
  br i1 %cmp8, label %if.then, label %if.end

if.then:
  %add = add i32 %3, %2
  %add12 = add i32 %add, %4
  %5 = ptrtoint ptr %0 to i32
  %add15 = add nsw i32 %1, %5
  %6 = inttoptr i32 %add15 to ptr
  %arrayidx = getelementptr inbounds i32, ptr %6, i32 %add12
  %7 = load i32, ptr %arrayidx, align 4
  br label %if.end

if.end:
  %sum.0 = phi i32 [ %7, %if.then ], [ 0, %land.lhs.true7 ], [ 0, %land.lhs.true4 ], [ 0, %land.lhs.true2 ], [ 0, %land.lhs.true ], [ 0, %entry ]
  ret i32 %sum.0
}

declare void @getnode(ptr sret(%struct.node_t), ...)
