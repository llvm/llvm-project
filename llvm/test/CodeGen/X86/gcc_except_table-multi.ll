; RUN: llc -simplifycfg-require-and-preserve-domtree=1 < %s -mtriple=x86_64 -binutils-version=2.35 | FileCheck %s --check-prefixes=CHECK,NORMAL
; RUN: llc -simplifycfg-require-and-preserve-domtree=1 < %s -mtriple=x86_64 -binutils-version=2.36 | FileCheck %s --check-prefixes=CHECK,NORMAL
; RUN: llc -simplifycfg-require-and-preserve-domtree=1 < %s -mtriple=x86_64 -function-sections -binutils-version=2.35 | FileCheck %s --check-prefixes=CHECK,SEP_BFD
; RUN: llc -simplifycfg-require-and-preserve-domtree=1 < %s -mtriple=x86_64 -function-sections -binutils-version=2.36 | FileCheck %s --check-prefixes=CHECK,SEP

; RUN: llc -simplifycfg-require-and-preserve-domtree=1 < %s -mtriple=x86_64 -function-sections -unique-section-names=false | FileCheck %s --check-prefixes=CHECK,SEP_NOUNIQUE
; RUN: llc -simplifycfg-require-and-preserve-domtree=1 < %s -mtriple=x86_64 -unique-section-names=false | FileCheck %s --check-prefixes=CHECK,NOUNIQUE

@_ZTIi = external constant i8*

;; If the function is in a comdat group, the generated .gcc_except_table should
;; be placed in the same group, so that .gcc_except_table can be discarded if
;; the comdat is not prevailing. If -funique-section-names, append the function name.
$group = comdat any
define i32 @group() uwtable comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL:       group:
; CHECK:             .cfi_endproc
; NORMAL-NEXT:       .section .gcc_except_table.group,"aG",@progbits,group,comdat{{$}}
; SEP_BFD-NEXT:      .section .gcc_except_table.group,"aG",@progbits,group,comdat{{$}}
; SEP-NEXT:          .section .gcc_except_table.group,"aGo",@progbits,group,comdat,group{{$}}
; SEP_NOUNIQUE-NEXT: .section .gcc_except_table,"aG",@progbits,group,comdat{{$}}
; NOUNIQUE-NEXT:     .section .gcc_except_table,"aG",@progbits,group,comdat{{$}}
entry:
  invoke void @ext() to label %try.cont unwind label %lpad
lpad:
  %0 = landingpad { i8*, i32 } catch i8* bitcast (i8** @_ZTIi to i8*)
  br label %eh.resume
try.cont:
  ret i32 0
eh.resume:
  resume { i8*, i32 } %0
}

;; If the function is not in a comdat group, but function sections is enabled,
;; use a separate section by either using a unique ID (integrated assembler) or
;; a suffix (GNU as<2.35).
define i32 @foo() uwtable personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL:       foo:
; CHECK:             .cfi_endproc
; NORMAL-NEXT:       .section .gcc_except_table,"a",@progbits{{$}}
; SEP_BFD-NEXT:      .section .gcc_except_table.foo,"a",@progbits{{$}}
; SEP-NEXT:          .section .gcc_except_table.foo,"ao",@progbits,foo{{$}}
; SEP_NOUNIQUE-NEXT: .section .gcc_except_table,"a",@progbits{{$}}
; NOUNIQUE-NEXT:     .section .gcc_except_table,"a",@progbits{{$}}
entry:
  invoke void @ext() to label %try.cont unwind label %lpad
lpad:
  %0 = landingpad { i8*, i32 } catch i8* bitcast (i8** @_ZTIi to i8*)
  br label %eh.resume
try.cont:
  ret i32 0
eh.resume:
  resume { i8*, i32 } %0
}

declare void @ext()

declare i32 @__gxx_personality_v0(...)
