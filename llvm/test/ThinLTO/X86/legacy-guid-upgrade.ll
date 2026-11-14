;; Test that ThinLTO handles bitcode produced before GUID assignment existed
;; (i.e. without a MODULE_CODE_GUIDLIST record and !guid metadata). GUIDs are
;; assigned on load from the original names and linkages, so the ThinLTO
;; backends can drop dead symbols, and promote and import across modules.
;;
;; The Inputs were produced from the modules below with an LLVM 21
;; opt -module-summary.

; RUN: llvm-lto2 run %p/Inputs/legacy-guid-upgrade-main.bc \
; RUN:   %p/Inputs/legacy-guid-upgrade-foo.bc -o %t -save-temps \
; RUN:   -r=%p/Inputs/legacy-guid-upgrade-main.bc,main,plx \
; RUN:   -r=%p/Inputs/legacy-guid-upgrade-main.bc,foo,l \
; RUN:   -r=%p/Inputs/legacy-guid-upgrade-foo.bc,foo,pl
; RUN: llvm-dis %t.1.3.import.bc -o - | FileCheck %s --check-prefix=MAIN
; RUN: llvm-dis %t.2.1.promote.bc -o - | FileCheck %s --check-prefix=FOO

;; The dead internal function is dropped (it would be silently retained if
;; its GUID were unavailable), and @foo is imported along with a declaration
;; of the promoted @bar.
; MAIN-NOT: @dead
; MAIN: define dso_local i32 @main() !guid ![[MAIN:[0-9]+]] {
; MAIN-NOT: @dead
; MAIN: define available_externally i32 @foo() !guid ![[IMPORTED:[0-9]+]] {
; MAIN-NOT: @dead
; MAIN: declare hidden i32 @bar.llvm.{{[0-9]+}}()
; MAIN-NOT: @dead
; MAIN: ![[MAIN]] = !{i64 -2624081020897602054}
; MAIN: ![[IMPORTED]] = !{i64 6699318081062747564}

;; Promoted local definitions keep the GUID computed from their original
;; name and linkage.
; FOO: @g = internal global i32 42, !guid ![[G:[0-9]+]]
; FOO: define dso_local i32 @foo() !guid ![[DEF:[0-9]+]] {
; FOO: define hidden i32 @bar.llvm.{{[0-9]+}}() {{.*}} !guid ![[BAR:[0-9]+]] {
; FOO: ![[G]] = !{i64 1417132069015513896}
; FOO: ![[DEF]] = !{i64 6699318081062747564}
; FOO: ![[BAR]] = !{i64 3742405216804959270}

;--- legacy-guid-upgrade-main.ll
source_filename = "legacy-thinlto-main.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
  %v = call i32 @foo()
  ret i32 %v
}

define internal void @dead() {
  ret void
}

declare i32 @foo()

;--- legacy-guid-upgrade-foo.ll
source_filename = "legacy-thinlto-foo.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g = internal global i32 42

define i32 @foo() {
  %v = call i32 @bar()
  ret i32 %v
}

define internal i32 @bar() noinline {
  %v = load i32, ptr @g
  ret i32 %v
}
