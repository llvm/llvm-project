;; Test callsite context graph generation for simple call graph with
;; two memprof contexts and no inlining, as well as graph and IR cloning.
;;
;; Original code looks like:
;;
;; char *bar() {
;;   return new char[10];
;; }
;;
;; char *baz() {
;;   return bar();
;; }
;;
;; char *foo() {
;;   return baz();
;; }
;;
;; int main(int argc, char **argv) {
;;   char *x = foo();
;;   char *y = foo();
;;   memset(x, 0, 10);
;;   memset(y, 0, 10);
;;   delete[] x;
;;   sleep(10);
;;   delete[] y;
;;   return 0;
;; }
;;
;; Code compiled with -mllvm -memprof-ave-lifetime-cold-threshold=5 so that the
;; memory freed after sleep(10) results in cold lifetimes.
;;
;; The IR was then reduced using llvm-reduce with the expected FileCheck input.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:	-memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:	-memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:	-stats -pass-remarks=memprof-context-disambiguation \
; RUN:	%s -S 2>&1 | FileCheck %s --check-prefix=DUMP --check-prefix=IR \
; RUN:	--check-prefix=STATS --check-prefix=REMARKS

; RUN:	cat %t.ccg.postbuild.dot | FileCheck %s --check-prefix=DOT
;; We should have cloned bar, baz, and foo, for the cold memory allocation.
; RUN:	cat %t.ccg.cloned.dot | FileCheck %s --check-prefix=DOTCLONED

;; Check again without -supports-hot-cold-new and ensure all MIB are cold and
;; that there is no cloning.
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:	-memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:	-memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:	-stats -pass-remarks=memprof-context-disambiguation \
; RUN:	%s -S 2>&1 | FileCheck %s --implicit-check-not="Callsite Context Graph" \
; RUN:	--implicit-check-not="created clone"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() #0 {
entry:
  %call = call noundef ptr @_Z3foov(), !callsite !0
  %call1 = call noundef ptr @_Z3foov(), !callsite !1
  ret i32 0
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #1

; Function Attrs: nobuiltin
declare void @_ZdaPv() #2

define internal ptr @_Z3barv() #3 {
entry:
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6, !memprof !2, !callsite !7
  ret ptr null
}

declare ptr @_Znam(i64)

define internal ptr @_Z3bazv() #4 {
entry:
  %call = call noundef ptr @_Z3barv(), !callsite !8
  ret ptr null
}

; Function Attrs: noinline
define internal ptr @_Z3foov() #5 {
entry:
  %call = call noundef ptr @_Z3bazv(), !callsite !9
  ret ptr null
}

; uselistorder directives
uselistorder ptr @_Z3foov, { 1, 0 }

attributes #0 = { "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { nobuiltin }
attributes #3 = { "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #4 = { "stack-protector-buffer-size"="8" }
attributes #5 = { noinline }
attributes #6 = { builtin }

!0 = !{i64 8632435727821051414}
!1 = !{i64 -3421689549917153178}
!2 = !{!3, !5}
!3 = !{!4, !"notcold"}
!4 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!5 = !{!6, !"cold"}
!6 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!7 = !{i64 9086428284934609951}
!8 = !{i64 -5964873800580613432}
!9 = !{i64 2732490490862098848}


; DUMP: CCG before cloning:
; DUMP: Callsite Context Graph:
; DUMP: Node [[BAR:0x[a-z0-9]+]]
; DUMP: 	  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[BAZ:0x[a-z0-9]+]] AllocTypes: NotColdCold ContextIds: 1 2

; DUMP: Node [[BAZ]]
; DUMP: 	  %call = call noundef ptr @_Z3barv()	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[BAZ]] AllocTypes: NotColdCold ContextIds: 1 2
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAZ]] to Caller: [[FOO:0x[a-z0-9]+]] AllocTypes: NotColdCold ContextIds: 1 2

; DUMP: Node [[FOO]]
; DUMP: 	  %call = call noundef ptr @_Z3bazv()	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAZ]] to Caller: [[FOO]] AllocTypes: NotColdCold ContextIds: 1 2
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 1
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN2:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 2

; DUMP: Node [[MAIN1]]
; DUMP: 	  %call = call noundef ptr @_Z3foov()	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN2]]
; DUMP: 	  %call1 = call noundef ptr @_Z3foov()	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:

; DUMP: CCG after cloning:
; DUMP: Callsite Context Graph:
; DUMP: Node [[BAR:0x[a-z0-9]+]]
; DUMP: 	  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[BAZ:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 1
; DUMP:		Clones: [[BAR2:0x[a-z0-9]+]]

; DUMP: Node [[BAZ]]
; DUMP: 	  %call = call noundef ptr @_Z3barv()	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR]] to Caller: [[BAZ]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAZ]] to Caller: [[FOO:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 1
; DUMP:		Clones: [[BAZ2:0x[a-z0-9]+]]

; DUMP: Node [[FOO]]
; DUMP: 	  %call = call noundef ptr @_Z3bazv()	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAZ]] to Caller: [[FOO]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1:0x[a-z0-9]+]] AllocTypes: NotCold ContextIds: 1
; DUMP:		Clones: [[FOO2:0x[a-z0-9]+]]

; DUMP: Node [[MAIN1]]
; DUMP: 	  %call = call noundef ptr @_Z3foov()	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN2]]
; DUMP: 	  %call1 = call noundef ptr @_Z3foov()	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:

; DUMP: Node [[FOO2]]
; DUMP: 	  %call = call noundef ptr @_Z3bazv()	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAZ2]] to Caller: [[FOO2]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 2
; DUMP:		Clone of [[FOO]]

; DUMP: Node [[BAZ2]]
; DUMP: 	  %call = call noundef ptr @_Z3barv()	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR2]] to Caller: [[BAZ2]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAZ2]] to Caller: [[FOO2]] AllocTypes: Cold ContextIds: 2
; DUMP:		Clone of [[BAZ]]

; DUMP: Node [[BAR2]]
; DUMP: 	  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR2]] to Caller: [[BAZ2]] AllocTypes: Cold ContextIds: 2
; DUMP:		Clone of [[BAR]]


; REMARKS: created clone _Z3barv.memprof.1
; REMARKS: created clone _Z3bazv.memprof.1
; REMARKS: created clone _Z3foov.memprof.1
; REMARKS: call in clone main assigned to call function clone _Z3foov.memprof.1
; REMARKS: call in clone _Z3foov.memprof.1 assigned to call function clone _Z3bazv.memprof.1
; REMARKS: call in clone _Z3bazv.memprof.1 assigned to call function clone _Z3barv.memprof.1
; REMARKS: call in clone _Z3barv.memprof.1 marked with memprof allocation attribute cold
; REMARKS: call in clone main assigned to call function clone _Z3foov
; REMARKS: call in clone _Z3foov assigned to call function clone _Z3bazv
; REMARKS: call in clone _Z3bazv assigned to call function clone _Z3barv
; REMARKS: call in clone _Z3barv marked with memprof allocation attribute notcold


; IR: define {{.*}} @main
;; The first call to foo does not allocate cold memory. It should call the
;; original functions, which ultimately call the original allocation decorated
;; with a "notcold" attribute.
; IR:   call {{.*}} @_Z3foov()
;; The second call to foo allocates cold memory. It should call cloned functions
;; which ultimately call a cloned allocation decorated with a "cold" attribute.
; IR:   call {{.*}} @_Z3foov.memprof.1()
; IR: define internal {{.*}} @_Z3barv()
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD:[0-9]+]]
; IR: define internal {{.*}} @_Z3bazv()
; IR:   call {{.*}} @_Z3barv()
; IR: define internal {{.*}} @_Z3foov()
; IR:   call {{.*}} @_Z3bazv()
; IR: define internal {{.*}} @_Z3barv.memprof.1()
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD:[0-9]+]]
; IR: define internal {{.*}} @_Z3bazv.memprof.1()
; IR:   call {{.*}} @_Z3barv.memprof.1()
; IR: define internal {{.*}} @_Z3foov.memprof.1()
; IR:   call {{.*}} @_Z3bazv.memprof.1()
; IR: attributes #[[NOTCOLD]] = { builtin "memprof"="notcold" }
; IR: attributes #[[COLD]] = { builtin "memprof"="cold" }


; STATS: 1 memprof-context-disambiguation - Number of cold static allocations (possibly cloned)
; STATS: 1 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned)
; STATS: 3 memprof-context-disambiguation - Number of function clones created during whole program analysis


; DOT: digraph "postbuild" {
; DOT: 	label="postbuild";
; DOT: 	Node[[BAR:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAR]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: Alloc0\n_Z3barv -\> _Znam}"];
; DOT: 	Node[[BAZ:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAZ]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 12481870273128938184\n_Z3bazv -\> _Z3barv}"];
; DOT: 	Node[[BAZ]] -> Node[[BAR]][tooltip="ContextIds: 1 2",fillcolor="mediumorchid1"];
; DOT: 	Node[[FOO:0x[a-z0-9]+]] [shape=record,tooltip="N[[FOO]] ContextIds: 1 2",fillcolor="mediumorchid1",style="filled",style="filled",label="{OrigId: 2732490490862098848\n_Z3foov -\> _Z3bazv}"];
; DOT: 	Node[[FOO]] -> Node[[BAZ]][tooltip="ContextIds: 1 2",fillcolor="mediumorchid1"];
; DOT: 	Node[[MAIN1:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN1]] ContextIds: 1",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 8632435727821051414\nmain -\> _Z3foov}"];
; DOT: 	Node[[MAIN1]] -> Node[[FOO]][tooltip="ContextIds: 1",fillcolor="brown1"];
; DOT: 	Node[[MAIN2:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN2]] ContextIds: 2",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 15025054523792398438\nmain -\> _Z3foov}"];
; DOT: 	Node[[MAIN2]] -> Node[[FOO]][tooltip="ContextIds: 2",fillcolor="cyan"];
; DOT: }


; DOTCLONED: digraph "cloned" {
; DOTCLONED: 	label="cloned";
; DOTCLONED: 	Node[[BAR:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAR]] ContextIds: 1",fillcolor="brown1",style="filled",style="filled",label="{OrigId: Alloc0\n_Z3barv -\> _Znam}"];
; DOTCLONED: 	Node[[BAZ:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAZ]] ContextIds: 1",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 12481870273128938184\n_Z3bazv -\> _Z3barv}"];
; DOTCLONED: 	Node[[BAZ]] -> Node[[BAR]][tooltip="ContextIds: 1",fillcolor="brown1"];
; DOTCLONED: 	Node[[FOO:0x[a-z0-9]+]] [shape=record,tooltip="N[[FOO]] ContextIds: 1",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 2732490490862098848\n_Z3foov -\> _Z3bazv}"];
; DOTCLONED: 	Node[[FOO]] -> Node[[BAZ]][tooltip="ContextIds: 1",fillcolor="brown1"];
; DOTCLONED: 	Node[[MAIN1:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN1]] ContextIds: 1",fillcolor="brown1",style="filled",style="filled",label="{OrigId: 8632435727821051414\nmain -\> _Z3foov}"];
; DOTCLONED: 	Node[[MAIN1]] -> Node[[FOO]][tooltip="ContextIds: 1",fillcolor="brown1"];
; DOTCLONED: 	Node[[MAIN2:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN2]] ContextIds: 2",fillcolor="cyan",style="filled",style="filled",label="{OrigId: 15025054523792398438\nmain -\> _Z3foov}"];
; DOTCLONED: 	Node[[MAIN2]] -> Node[[FOO2:0x[a-z0-9]+]][tooltip="ContextIds: 2",fillcolor="cyan"];
; DOTCLONED: 	Node[[FOO2]] [shape=record,tooltip="N[[FOO2]] ContextIds: 2",fillcolor="cyan",style="filled",color="blue",style="filled,bold,dashed",label="{OrigId: 0\n_Z3foov -\> _Z3bazv}"];
; DOTCLONED: 	Node[[FOO2]] -> Node[[BAZ2:0x[a-z0-9]+]][tooltip="ContextIds: 2",fillcolor="cyan"];
; DOTCLONED: 	Node[[BAZ2]] [shape=record,tooltip="N[[BAZ2]] ContextIds: 2",fillcolor="cyan",style="filled",color="blue",style="filled,bold,dashed",label="{OrigId: 0\n_Z3bazv -\> _Z3barv}"];
; DOTCLONED: 	Node[[BAZ2]] -> Node[[BAR2:0x[a-z0-9]+]][tooltip="ContextIds: 2",fillcolor="cyan"];
; DOTCLONED: 	Node[[BAR2]] [shape=record,tooltip="N[[BAR2]] ContextIds: 2",fillcolor="cyan",style="filled",color="blue",style="filled,bold,dashed",label="{OrigId: Alloc0\n_Z3barv -\> _Znam}"];
; DOTCLONED: }
