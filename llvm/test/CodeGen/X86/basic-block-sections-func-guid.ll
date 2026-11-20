;; Test matching functions with local linkage by their GUIDs ('u' specifier) in
;; the basic block sections profile.
;;
;; @work and @foo have local linkage and @bar has external linkage. The GUID of
;; @work is computed from the source file name and the function name, i.e.,
;; MD5("a.c;work") = 0x776493a8cf165896, as emitted in the BB address map.
; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llc < test.ll -mtriple=x86_64 -function-sections -basic-block-address-map \
; RUN:   -basic-block-address-map-emit-func-guid | FileCheck %s --check-prefix=GUID
; GUID: .quad 8603163541422626966 # function GUID

;; A function with a profile is placed in its .text.hot section, while a
;; function without a profile stays in its .text section. The profiles which
;; must apply specify the single cluster "c 0", which also moves the other
;; blocks of the function to its .text.split section. The profile for @work
;; with a different GUID places all blocks in one cluster ("c 0 1 2"), which
;; doesn't create a .text.split section.

;; With function GUIDs, only the profile with the matching GUID applies to
;; @work. Profiles without GUIDs only apply to functions with non-local linkage
;; (@bar), since the profile must specify the GUIDs of all local-linkage
;; functions.
; RUN: llc < test.ll -mtriple=x86_64 -function-sections -basic-block-sections=guids.txt | \
; RUN:   FileCheck %s --check-prefixes=CHECK,NO-FOO
;; Without function GUIDs, profiles apply to all functions with the same names.
; RUN: llc < test.ll -mtriple=x86_64 -function-sections -basic-block-sections=no-guids.txt | \
; RUN:   FileCheck %s --check-prefixes=CHECK,FOO

; CHECK:  .section .text.split.work,"ax",@progbits
; NO-FOO: .section .text.foo,"ax",@progbits
; FOO:    .section .text.split.foo,"ax",@progbits
; CHECK:  .section .text.split.bar,"ax",@progbits

;--- guids.txt
v1
u 1234
f work
c 0 1 2
u 776493a8cf165896
f work
c 0
f foo
c 0
f bar
c 0
;--- no-guids.txt
v1
f work
c 0
f foo
c 0
f bar
c 0
;--- test.ll
source_filename = "a.c"

define internal i32 @work(i32 %x) {
entry:
  %cmp = icmp slt i32 %x, 0
  br i1 %cmp, label %if.then, label %if.else
if.then:
  ret i32 0
if.else:
  ret i32 1
}

define internal i32 @foo(i32 %x) {
entry:
  %cmp = icmp slt i32 %x, 0
  br i1 %cmp, label %if.then, label %if.else
if.then:
  ret i32 0
if.else:
  ret i32 1
}

define i32 @bar(i32 %x) {
entry:
  %cmp = icmp slt i32 %x, 0
  br i1 %cmp, label %if.then, label %if.else
if.then:
  ret i32 0
if.else:
  ret i32 1
}
