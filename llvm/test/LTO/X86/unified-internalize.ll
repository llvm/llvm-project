; RUN: opt <%s -unified-lto -thinlto-split-lto-unit -thinlto-bc -o %t.bc

; Test internalization during unified LTO. This makes sure internalization does
; happen in runRegularLTO().
; RUN: llvm-lto2 run %t.bc -o %t.o -save-temps --unified-lto=full \
; RUN:     -r=%t.bc,salad,pxl \
; RUN:     -r=%t.bc,balsamic,pl \
; RUN:     -r=%t.bc,thousandisland,pl \
; RUN:     -r=%t.bc,main,pxl \
; RUN:     -r %t.bc,ranch,px \
; RUN:     -r %t.bc,egg, \
; RUN:     -r %t.bc,bar,px
; RUN: llvm-dis < %t.o.0.2.internalize.bc | FileCheck  %s

; CHECK: @llvm.used = appending global {{.*}} @bar
; CHECK: define dso_local dllexport void @thousandisland
; CHECK: define dso_local void @salad
; CHECK: define internal void @balsamic
; CHECK: define dso_local void @main
; CHECK: define available_externally void @egg()

target triple = "x86_64-scei-ps4"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @salad() {
    call void @balsamic()
    ret void
}
define void @balsamic() {
    ret void
}
define dllexport void @thousandisland() {
    ret void
}

define void @main() {
    ret void
}

define void ()* @ranch() {
  ret void ()* @egg
}

define available_externally void @egg() {
  ret void
}

%"foo.1" = type { i8, i8 }
declare dso_local i32 @bar(%"foo.1"* nocapture readnone %this) local_unnamed_addr
@llvm.used = appending global [2 x i8*] [i8* bitcast (i32 (%"foo.1"*)* @bar to i8*), i8* bitcast (void ()* @thousandisland to i8*)], section "llvm.metadata"
