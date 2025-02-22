;; Test callsite context graph exporting to dot with various options.
;;
;; The code is similar to that of basic.ll, but with a second allocation.

;; Check expected error if alloc scope requested without an alloc id.
; RUN: not --crash opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:	-memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:	-memprof-dot-scope=alloc \
; RUN:	%s -S 2>&1 | FileCheck %s --check-prefix=ERRMISSINGALLOCID
; ERRMISSINGALLOCID: -memprof-dot-scope=alloc requires -memprof-dot-alloc-id

;; Check expected error if context scope requested without a context id.
; RUN: not --crash opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:	-memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:	-memprof-dot-scope=context \
; RUN:	%s -S 2>&1 | FileCheck %s --check-prefix=ERRMISSINGCONTEXTID
; ERRMISSINGCONTEXTID: -memprof-dot-scope=context requires -memprof-dot-context-id

;; Check expected error if both alloc and context ids are specified when we are
;; exporting the full graph to dot. It is unclear which scope to highlight.
; RUN: not --crash opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:	-memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:	-memprof-dot-alloc-id=0 -memprof-dot-context-id=2 \
; RUN:	%s -S 2>&1 | FileCheck %s --check-prefix=ERRBOTH
; ERRBOTH: -memprof-dot-scope=all can't have both -memprof-dot-alloc-id and -memprof-dot-context-id

;; Export full graph (default), without any highlight.
; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:	-memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:	%s -S 2>&1 | FileCheck %s --check-prefix=IR
; RUN:	cat %t.ccg.postbuild.dot | FileCheck %s -check-prefix=DOTCOMMON --check-prefix=DOTALLANDALLOC0 --check-prefix=DOTALL --check-prefix=DOTALLNONE

;; Export full graph (default), with highlight of alloc 0.
; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:	-memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:	-memprof-dot-alloc-id=0 \
; RUN:	%s -S 2>&1 | FileCheck %s --check-prefix=IR
; RUN:	cat %t.ccg.postbuild.dot | FileCheck %s --check-prefix=DOTCOMMON --check-prefix=DOTALLANDALLOC0 --check-prefix=DOTALL --check-prefix=DOTALLALLOC0

;; Export full graph (default), with highlight of context 1.
; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:	-memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:	-memprof-dot-context-id=1 \
; RUN:	%s -S 2>&1 | FileCheck %s --check-prefix=IR
; RUN:	cat %t.ccg.postbuild.dot | FileCheck %s --check-prefix=DOTCOMMON --check-prefix=DOTALLANDALLOC0 --check-prefix=DOTALL --check-prefix=DOTALLCONTEXT1

;; Export alloc 0 only, without any highlight.
; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:	-memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:	-memprof-dot-scope=alloc -memprof-dot-alloc-id=0 \
; RUN:	%s -S 2>&1 | FileCheck %s --check-prefix=IR
; RUN:	cat %t.ccg.postbuild.dot | FileCheck %s --check-prefix=DOTCOMMON --check-prefix=DOTALLANDALLOC0 --check-prefix=DOTALLOC0NONE

;; Export alloc 0 only, with highlight of context 1.
; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:	-memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:	-memprof-dot-scope=alloc -memprof-dot-alloc-id=0 \
; RUN:	-memprof-dot-context-id=1 \
; RUN:	%s -S 2>&1 | FileCheck %s --check-prefix=IR
; RUN:	cat %t.ccg.postbuild.dot | FileCheck %s --check-prefix=DOTCOMMON --check-prefix=DOTALLANDALLOC0 --check-prefix=DOTALLOC0CONTEXT1

;; Export context 1 only (which means no highlighting).
; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:	-memprof-export-to-dot -memprof-dot-file-path-prefix=%t. \
; RUN:	-memprof-dot-scope=context -memprof-dot-context-id=1 \
; RUN:	%s -S 2>&1 | FileCheck %s --check-prefix=IR
; RUN:	cat %t.ccg.postbuild.dot | FileCheck %s --check-prefix=DOTCOMMON --check-prefix=DOTCONTEXT1

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
  %call2 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6, !memprof !13, !callsite !18
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
!3 = !{!4, !"notcold", !10}
!4 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!5 = !{!6, !"cold", !11, !12}
!6 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!7 = !{i64 9086428284934609951}
!8 = !{i64 -5964873800580613432}
!9 = !{i64 2732490490862098848}
!10 = !{i64 123, i64 100}
!11 = !{i64 456, i64 200}
!12 = !{i64 789, i64 300}
!13 = !{!14, !16}
!14 = !{!15, !"notcold", !10}
!15 = !{i64 123, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!16 = !{!17, !"cold", !11, !12}
!17 = !{i64 123, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!18 = !{i64 123}

; IR: define {{.*}} @main
;; The first call to foo does not allocate cold memory. It should call the
;; original functions, which ultimately call the original allocations decorated
;; with a "notcold" attribute.
; IR:   call {{.*}} @_Z3foov()
;; The second call to foo allocates cold memory. It should call cloned functions
;; which ultimately call acloned allocations decorated with a "cold" attribute.
; IR:   call {{.*}} @_Z3foov.memprof.1()
; IR: define internal {{.*}} @_Z3barv()
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD:[0-9]+]]
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD]]
; IR: define internal {{.*}} @_Z3bazv()
; IR:   call {{.*}} @_Z3barv()
; IR: define internal {{.*}} @_Z3foov()
; IR:   call {{.*}} @_Z3bazv()
; IR: define internal {{.*}} @_Z3barv.memprof.1()
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD:[0-9]+]]
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD]]
; IR: define internal {{.*}} @_Z3bazv.memprof.1()
; IR:   call {{.*}} @_Z3barv.memprof.1()
; IR: define internal {{.*}} @_Z3foov.memprof.1()
; IR:   call {{.*}} @_Z3bazv.memprof.1()
; IR: attributes #[[NOTCOLD]] = { builtin "memprof"="notcold" }
; IR: attributes #[[COLD]] = { builtin "memprof"="cold" }


;; Check highlighting. Alloc 0 includes context ids 1 and 2.

; DOTCOMMON:  Node[[BAR:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAR]] ContextIds: 1 2",
;; This node is highlighted when dumping the whole graph and specifying
;; alloc 0 or context 1, or if dumping alloc 0 and specifying context 1.
; DOTALLNONE-SAME: fillcolor="mediumorchid1",
; DOTALLALLOC0-SAME: fontsize="30",fillcolor="magenta",
; DOTALLCONTEXT1-SAME: fontsize="30",fillcolor="magenta",
; DOTALLOC0NONE-SAME: fillcolor="mediumorchid1",
; DOTALLOC0CONTEXT1-SAME: fontsize="30",fillcolor="magenta",
; DOTCONTEXT1-SAME: fillcolor="mediumorchid1",
; DOTCOMMON-SAME: style="filled",label="{OrigId: Alloc0\n_Z3barv -\> _Znam}"];

; DOTCOMMON:  Node[[BAZ:0x[a-z0-9]+]] [shape=record,tooltip="N[[BAZ]] ContextIds: 1 2 3 4",
;; This node is highlighted when dumping the whole graph and specifying
;; alloc 0 or context 1, or if dumping alloc 0 and specifying context 1.
; DOTALLNONE-SAME: fillcolor="mediumorchid1",
; DOTALLALLOC0-SAME: fontsize="30",fillcolor="magenta",
; DOTALLCONTEXT1-SAME: fontsize="30",fillcolor="magenta",
; DOTALLOC0NONE-SAME: fillcolor="mediumorchid1",
; DOTALLOC0CONTEXT1-SAME: fontsize="30",fillcolor="magenta",
; DOTCONTEXT1-SAME: fillcolor="mediumorchid1",
; DOTCOMMON-SAME: style="filled",label="{OrigId: 12481870273128938184\n_Z3bazv -\> _Z3barv}"];

; DOTCOMMON:  Node[[BAZ]] -> Node[[BAR]][tooltip="ContextIds: 1 2",
;; This edge is highlighted when dumping the whole graph and specifying
;; alloc 0 or context 1, or if dumping alloc 0 and specifying context 1.
; DOTALLNONE-SAME: fillcolor="mediumorchid1",color="mediumorchid1"];
; DOTALLALLOC0-SAME: fillcolor="magenta",color="magenta",penwidth="2.0",weight="2"];
; DOTALLCONTEXT1-SAME: fillcolor="magenta",color="magenta",penwidth="2.0",weight="2"];
; DOTALLOC0NONE-SAME: fillcolor="mediumorchid1",color="mediumorchid1"];
; DOTALLOC0CONTEXT1-SAME: fillcolor="magenta",color="magenta",penwidth="2.0",weight="2"];
; DOTCONTEXT1-SAME: fillcolor="mediumorchid1",color="mediumorchid1"];

;; This edge is not in alloc 0 or context 0, so only included when exporting
;; the whole graph (and never highlighted).
; DOTALL:  Node[[BAZ]] -> Node[[BAR2:0x[a-z0-9]+]][tooltip="ContextIds: 3 4",fillcolor="mediumorchid1",color="mediumorchid1"];

; DOTCOMMON:  Node[[FOO:0x[a-z0-9]+]] [shape=record,tooltip="N[[FOO]] ContextIds: 1 2 3 4",
;; This node is highlighted when dumping the whole graph and specifying
;; alloc 0 or context 1, or if dumping alloc 0 and specifying context 1.
; DOTALLNONE-SAME: fillcolor="mediumorchid1",
; DOTALLALLOC0-SAME: fontsize="30",fillcolor="magenta",
; DOTALLCONTEXT1-SAME: fontsize="30",fillcolor="magenta",
; DOTALLOC0NONE-SAME: fillcolor="mediumorchid1",
; DOTALLOC0CONTEXT1-SAME: fontsize="30",fillcolor="magenta",
; DOTCONTEXT1-SAME: fillcolor="mediumorchid1",
; DOTCOMMON-SAME: style="filled",label="{OrigId: 2732490490862098848\n_Z3foov -\> _Z3bazv}"];

; DOTCOMMON:  Node[[FOO]] -> Node[[BAZ]][tooltip="ContextIds: 1 2 3 4",
;; This edge is highlighted when dumping the whole graph and specifying
;; alloc 0 or context 1, or if dumping alloc 0 and specifying context 1.
; DOTALLNONE-SAME: fillcolor="mediumorchid1",color="mediumorchid1"];
; DOTALLALLOC0-SAME: fillcolor="magenta",color="magenta",penwidth="2.0",weight="2"];
; DOTALLCONTEXT1-SAME: fillcolor="magenta",color="magenta",penwidth="2.0",weight="2"];
; DOTALLOC0NONE-SAME: fillcolor="mediumorchid1",color="mediumorchid1"];
; DOTALLOC0CONTEXT1-SAME: fillcolor="magenta",color="magenta",penwidth="2.0",weight="2"];
; DOTCONTEXT1-SAME: fillcolor="mediumorchid1",color="mediumorchid1"];

; DOTCOMMON:  Node[[MAIN1:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN1]] ContextIds: 1 3",
;; This node is highlighted when dumping the whole graph and specifying
;; alloc 0 or context 1, or if dumping alloc 0 and specifying context 1.
;; Note that the highlight color is the same as when there is no highlighting.
; DOTALLALLOC0-SAME: fontsize="30",
; DOTALLCONTEXT1-SAME: fontsize="30",
; DOTALLOC0CONTEXT1-SAME: fontsize="30",
; DOTCOMMON-SAME: fillcolor="brown1",style="filled",label="{OrigId: 8632435727821051414\nmain -\> _Z3foov}"];

; DOTCOMMON:  Node[[MAIN1]] -> Node[[FOO]][tooltip="ContextIds: 1 3",fillcolor="brown1",color="brown1"
;; This edge is highlighted when dumping the whole graph and specifying
;; alloc 0 or context 1, or if dumping alloc 0 and specifying context 1.
; DOTALLNONE-SAME: ];
; DOTALLALLOC0-SAME: penwidth="2.0",weight="2"];
; DOTALLCONTEXT1-SAME: penwidth="2.0",weight="2"];
; DOTALLOC0NONE-SAME: ];
; DOTALLOC0CONTEXT1-SAME: penwidth="2.0",weight="2"];
; DOTCONTEXT1-SAME: ];

; DOTALLANDALLOC0:  Node[[MAIN2:0x[a-z0-9]+]] [shape=record,tooltip="N[[MAIN2]] ContextIds: 2 4",
;; This node is highlighted when dumping the whole graph and specifying
;; alloc 0. Note that the unhighlighted color is different when there is any
;; highlighting (lightskyblue) vs no highlighting (cyan).
; DOTALLNONE-SAME: fillcolor="cyan",
; DOTALLALLOC0-SAME: fontsize="30",fillcolor="cyan",
; DOTALLCONTEXT1-SAME: fillcolor="lightskyblue",
; DOTALLOC0NONE-SAME: fillcolor="cyan",
; DOTALLOC0CONTEXT1-SAME: fillcolor="lightskyblue",
; DOTALLANDALLOC0-SAME: style="filled",label="{OrigId: 15025054523792398438\nmain -\> _Z3foov}"];

; DOTALLANDALLOC0:  Node[[MAIN2]] -> Node[[FOO]][tooltip="ContextIds: 2 4",
;; This edge is highlighted when dumping the whole graph and specifying
;; alloc 0. Note that the unhighlighted color is different when there is any
;; highlighting (lightskyblue) vs no highlighting (cyan).
; DOTALLNONE-SAME: fillcolor="cyan",color="cyan"];
; DOTALLALLOC0-SAME: fillcolor="cyan",color="cyan",penwidth="2.0",weight="2"];
; DOTALLCONTEXT1-SAME: fillcolor="lightskyblue",color="lightskyblue"];
; DOTALLOC0NONE-SAME: fillcolor="cyan",color="cyan"];
; DOTALLOC0CONTEXT1-SAME: fillcolor="lightskyblue",color="lightskyblue"];

;; This edge is not in alloc 0 or context 0, so only included when exporting
;; the whole graph (and never highlighted).
; DOTALL:  Node[[BAR2]] [shape=record,tooltip="N[[BAR2]] ContextIds: 3 4",fillcolor="mediumorchid1",style="filled",label="{OrigId: Alloc2\n_Z3barv -\> _Znam}"];
