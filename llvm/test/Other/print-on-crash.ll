; A test that the hidden option -print-on-crash properly sets a signal handler
; which gets called when a pass crashes.  The trigger-crash pass asserts.

; RUN: not --crash opt -print-on-crash -passes=trigger-crash < %s 2>&1 | FileCheck %s --check-prefix=CHECK_SIMPLE

; A test that the signal handler set by the  hidden option -print-on-crash
; is not called when no pass crashes.

; RUN: opt -print-on-crash -passes="default<O2>" < %s 2>&1 | FileCheck %s --check-prefix=CHECK_NO_CRASH

; RUN: not --crash opt -print-on-crash -print-module-scope -passes=trigger-crash < %s 2>&1 | FileCheck %s --check-prefix=CHECK_MODULE

; RUN: not --crash opt -print-on-crash -print-module-scope -passes=trigger-crash -filter-passes=TriggerCrashPass < %s 2>&1 | FileCheck %s --check-prefix=CHECK_MODULE

; RUN: not --crash opt -print-on-crash -print-module-scope -passes=trigger-crash -filter-passes=blah < %s 2>&1 | FileCheck %s --check-prefix=CHECK_FILTERED

; CHECK_SIMPLE: *** Dump of IR Before Last Pass {{.*}} Started ***
; CHECK_SIMPLE: @main
; CHECK_SIMPLE: entry:
; CHECK_NO_CRASH-NOT: *** Dump of IR
; CHECK_MODULE: *** Dump of Module IR Before Last Pass {{.*}} Started ***
; CHECK_MODULE: ; ModuleID = {{.*}}
; CHECK_FILTERED: *** Dump of Module IR Before Last Pass {{.*}} Filtered Out ***

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  ret i32 0
}
