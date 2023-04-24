;; Test callsite context graph generation for call graph with with MIBs
;; that have pruned contexts that partially match multiple inlined
;; callsite contexts, requiring duplication of context ids and nodes
;; while matching callsite nodes onto the graph.
;;
;; Original code looks like:
;;
;; char *D() {
;;   return new char[10];
;; }
;;
;; char *F() {
;;   return D();
;; }
;;
;; char *C() {
;;   return D();
;; }
;;
;; char *B() {
;;   return C();
;; }
;;
;; char *E() {
;;   return C();
;; }
;; int main(int argc, char **argv) {
;;   char *x = B(); // cold
;;   char *y = E(); // cold
;;   char *z = F(); // default
;;   memset(x, 0, 10);
;;   memset(y, 0, 10);
;;   memset(z, 0, 10);
;;   delete[] z;
;;   sleep(10);
;;   delete[] x;
;;   delete[] y;
;;   return 0;
;; }
;;
;; Code compiled with -mllvm -memprof-min-lifetime-cold-threshold=5 so that the
;; memory freed after sleep(10) results in cold lifetimes.
;;
;; The code below was created by forcing inlining of C into both B and E.
;; Since both allocation contexts via C are cold, the matched memprof
;; metadata has the context pruned above C's callsite. This requires
;; matching the stack node for C to callsites where it was inlined (i.e.
;; the callsites in B and E that have callsite metadata that includes C's).
;; It also requires duplication of that node in the graph as well as the
;; duplication of the context ids along that path through the graph,
;; so that we can represent the duplicated (via inlining) C callsite.
;;
;; The IR was then reduced using llvm-reduce with the expected FileCheck input.

; RUN: opt -passes=memprof-context-disambiguation \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:  -memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:  %s -S 2>&1 | FileCheck %s --check-prefix=DUMP

; RUN:  cat %t.ccg.prestackupdate.dot | FileCheck %s --check-prefix=DOTPRE
; RUN:  cat %t.ccg.postbuild.dot | FileCheck %s --check-prefix=DOTPOST

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal ptr @_Z1Dv() {
entry:
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6, !memprof !0, !callsite !5
  ret ptr null
}

declare ptr @_Znam(i64)

define internal ptr @_Z1Fv() #0 {
entry:
  %call = call noundef ptr @_Z1Dv(), !callsite !6
  ret ptr null
}

; Function Attrs: mustprogress noinline optnone uwtable
define internal ptr @_Z1Cv() #1 {
entry:
  %call = call noundef ptr @_Z1Dv(), !callsite !7
  ret ptr null
}

; Function Attrs: mustprogress noinline optnone uwtable
define internal ptr @_Z1Bv() #1 {
entry:
  %call.i = call noundef ptr @_Z1Dv(), !callsite !8
  ret ptr null
}

; Function Attrs: mustprogress noinline optnone uwtable
define internal ptr @_Z1Ev() #1 {
entry:
  %call.i = call noundef ptr @_Z1Dv(), !callsite !9
  ret ptr null
}

; Function Attrs: noinline
declare i32 @main() #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #3

; Function Attrs: nounwind
declare void @_ZdaPv() #4

declare i32 @sleep() #5

attributes #0 = { "disable-tail-calls"="true" }
attributes #1 = { mustprogress noinline optnone uwtable "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { noinline }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #4 = { nounwind }
attributes #5 = { "no-trapping-math"="true" }
attributes #6 = { builtin }

!0 = !{!1, !3}
!1 = !{!2, !"cold"}
!2 = !{i64 6541423618768552252, i64 -6270142974039008131}
!3 = !{!4, !"notcold"}
!4 = !{i64 6541423618768552252, i64 -4903163940066524832}
!5 = !{i64 6541423618768552252}
!6 = !{i64 -4903163940066524832}
!7 = !{i64 -6270142974039008131}
!8 = !{i64 -6270142974039008131, i64 -184525619819294889}
!9 = !{i64 -6270142974039008131, i64 1905834578520680781}


;; After adding only the alloc node memprof metadata, we only have 2 contexts.

; DUMP: CCG before updating call stack chains:
; DUMP: Callsite Context Graph:
; DUMP: Node [[D:0x[a-z0-9]+]]
; DUMP: 	  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[D]] to Caller: [[C:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 1
; DUMP: 		Edge from Callee [[D]] to Caller: [[F:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 2

; DUMP: Node [[C]]
; DUMP: 	null Call
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[D]] to Caller: [[C]] AllocTypes: Cold ContextIds: 1
; DUMP: 	CallerEdges:

; DUMP: Node [[F]]
; DUMP: 	null Call
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[D]] to Caller: [[F]] AllocTypes: NotCold ContextIds: 2
; DUMP: 	CallerEdges:

;; After updating for callsite metadata, we should have generated context ids 3 and 4,
;; along with 2 new nodes for those callsites. All have the same allocation type
;; behavior as the original C node.

; DUMP: CCG before cloning:
; DUMP: Callsite Context Graph:
; DUMP: Node [[D]]
; DUMP: 	  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2 3 4
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[D]] to Caller: [[F]] AllocTypes: NotCold ContextIds: 2
; DUMP: 		Edge from Callee [[D]] to Caller: [[C2:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 3
; DUMP: 		Edge from Callee [[D]] to Caller: [[B:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 4
; DUMP: 		Edge from Callee [[D]] to Caller: [[E:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 1

; DUMP: Node [[F]]
; DUMP: 	  %call = call noundef ptr @_Z1Dv()	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[D]] to Caller: [[F]] AllocTypes: NotCold ContextIds: 2
; DUMP: 	CallerEdges:

; DUMP: Node [[C2]]
; DUMP: 	  %call = call noundef ptr @_Z1Dv()	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[D]] to Caller: [[C2]] AllocTypes: Cold ContextIds: 3
; DUMP: 	CallerEdges:

; DUMP: Node [[B]]
; DUMP: 	  %call.i = call noundef ptr @_Z1Dv()	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[D]] to Caller: [[B]] AllocTypes: Cold ContextIds: 4
; DUMP: 	CallerEdges:

; DUMP: Node [[E]]
; DUMP: 	  %call.i = call noundef ptr @_Z1Dv()	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[D]] to Caller: [[E]] AllocTypes: Cold ContextIds: 1
; DUMP: 	CallerEdges:


; DOTPRE: digraph "prestackupdate" {
; DOTPRE: 	label="prestackupdate";
; DOTPRE: 	Node[[D:0x[a-z0-9]+]] [shape=record,tooltip="N[[D]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: Alloc0\n_Z1Dv -\> _Znam}"];
; DOTPRE: 	Node[[C:0x[a-z0-9]+]] [shape=record,tooltip="N[[C]] ContextIds: 1",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 12176601099670543485\nnull call (external)}"];
; DOTPRE: 	Node[[C]] -> Node[[D]][tooltip="ContextIds: 1",fillcolor="cyan"];
; DOTPRE: 	Node[[F:0x[a-z0-9]+]] [shape=record,tooltip="N[[F]] ContextIds: 2",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 13543580133643026784\nnull call (external)}"];
; DOTPRE: 	Node[[F]] -> Node[[D]][tooltip="ContextIds: 2",fillcolor="brown1"];
; DOTPRE: }


; DOTPOST:digraph "postbuild" {
; DOTPOST:	label="postbuild";
; DOTPOST:	Node[[D:0x[a-z0-9]+]] [shape=record,tooltip="N[[D]] ContextIds: 1 2 3 4",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: Alloc0\n_Z1Dv -\> _Znam}"];
; DOTPOST:	Node[[F:0x[a-z0-9]+]] [shape=record,tooltip="N[[F]] ContextIds: 2",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 13543580133643026784\n_Z1Fv -\> _Z1Dv}"];
; DOTPOST:	Node[[F]] -> Node[[D]][tooltip="ContextIds: 2",fillcolor="brown1"];
; DOTPOST:	Node[[C:0x[a-z0-9]+]] [shape=record,tooltip="N[[C]] ContextIds: 3",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 0\n_Z1Cv -\> _Z1Dv}"];
; DOTPOST:	Node[[C]] -> Node[[D]][tooltip="ContextIds: 3",fillcolor="cyan"];
; DOTPOST:	Node[[B:0x[a-z0-9]+]] [shape=record,tooltip="N[[B]] ContextIds: 4",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 0\n_Z1Bv -\> _Z1Dv}"];
; DOTPOST:	Node[[B]] -> Node[[D]][tooltip="ContextIds: 4",fillcolor="cyan"];
; DOTPOST:	Node[[E:0x[a-z0-9]+]] [shape=record,tooltip="N[[E]] ContextIds: 1",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 0\n_Z1Ev -\> _Z1Dv}"];
; DOTPOST:	Node[[E]] -> Node[[D]][tooltip="ContextIds: 1",fillcolor="cyan"];
; DOTPOST:}
