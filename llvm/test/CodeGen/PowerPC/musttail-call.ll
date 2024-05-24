; RUN: not --crash llc -mtriple=powerpc64-ibm-aix-xcoff %s -o - 2>&1 -ppc-abort-on-impossible-musttailcall=true | \
; RUN:   FileCheck %s --check-prefix=CRASH
; RUN: not --crash llc -mtriple=powerpc-ibm-aix-xcoff %s -o - 2>&1 -ppc-abort-on-impossible-musttailcall=true | \
; RUN:   FileCheck %s --check-prefix=CRASH
; RUN: not --crash llc -mtriple=powerpc64-unknown-linux-gnu %s -o - 2>&1 -ppc-abort-on-impossible-musttailcall=true | \
; RUN:   FileCheck %s --check-prefix=CRASH
; RUN: not --crash llc -mtriple=powerpc-unknown-linux-gnu %s -o - 2>&1 -ppc-abort-on-impossible-musttailcall=true | \
; RUN:   FileCheck %s --check-prefix=CRASH
; RUN: not --crash llc -mtriple=powerpc64le-unknown-linux-gnu %s -o - 2>&1 -ppc-abort-on-impossible-musttailcall=true | \
; RUN:   FileCheck %s --check-prefix=CRASH
; RUN: llc -mtriple=powerpc64-ibm-aix-xcoff %s -o - 2>&1 -ppc-abort-on-impossible-musttailcall=false | \
; RUN:   FileCheck %s --check-prefix=NOCRASH
; RUN: llc -mtriple=powerpc-ibm-aix-xcoff %s -o - 2>&1 -ppc-abort-on-impossible-musttailcall=false | \
; RUN:   FileCheck %s --check-prefix=NOCRASH
; RUN: llc -mtriple=powerpc64-unknown-linux-gnu %s -o - 2>&1 -ppc-abort-on-impossible-musttailcall=false | \
; RUN:   FileCheck %s --check-prefix=NOCRASH
; RUN: llc -mtriple=powerpc-unknown-linux-gnu %s -o - 2>&1 -ppc-abort-on-impossible-musttailcall=false | \
; RUN:   FileCheck %s --check-prefix=NOCRASH
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu %s -o - 2>&1 -ppc-abort-on-impossible-musttailcall=false | \
; RUN:   FileCheck %s --check-prefix=NOCRASH

; CRASH: LLVM ERROR: failed to perform tail call elimination on a call site marked musttail
; NOCRASH-NOT: LLVM ERROR: failed to perform tail call elimination on a call site marked musttail
; NOCRASH-LABEL: caller
; NOCRASH:    bl {{callee|.callee}}


declare i32 @callee()
define i32 @caller() {
  %res = musttail call i32 @callee()
  ret i32 %res
}
