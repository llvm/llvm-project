; REQUIRES: asserts
; Check that in ThinLTO pre-link the sample profile reader also loads the
; profiles of top-level functions that contain an inlined instance of a
; function defined in the module, using the inlinee-to-top-level map section.
; bar is inlined into main (via foo) and baz, and has no profile of its own.

; RUN: llvm-profdata merge --sample --extbinary --gen-inlinee-top-level-map %S/Inputs/inlinee-top-level.prof -o %t.afdo
; RUN: llvm-profdata merge --sample --extbinary --gen-inlinee-top-level-map --use-md5 %S/Inputs/inlinee-top-level.prof -o %t.md5.afdo
; RUN: llvm-profdata merge --sample --extbinary %S/Inputs/inlinee-top-level.prof -o %t.nomap.afdo

; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%t.afdo -debug-only=samplepgo-reader -o /dev/null 2>&1 | FileCheck %s --check-prefix=PRELINK
; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%t.md5.afdo -debug-only=samplepgo-reader -o /dev/null 2>&1 | FileCheck %s --check-prefix=PRELINK-MD5

; Not loaded when disabled, outside ThinLTO pre-link, or without the section.
; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%t.afdo -sample-profile-load-inlinee-top-level=0 -debug-only=samplepgo-reader -o /dev/null 2>&1 | FileCheck %s --allow-empty --check-prefix=NOLOAD
; RUN: opt < %s -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%t.afdo -debug-only=samplepgo-reader -o /dev/null 2>&1 | FileCheck %s --allow-empty --check-prefix=NOLOAD
; RUN: opt < %s -passes='default<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%t.afdo -debug-only=samplepgo-reader -o /dev/null 2>&1 | FileCheck %s --allow-empty --check-prefix=NOLOAD
; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%t.nomap.afdo -debug-only=samplepgo-reader -o /dev/null 2>&1 | FileCheck %s --allow-empty --check-prefix=NOLOAD

; Profiles are loaded in GUID order; qux contains no inlinee from this module.
; PRELINK-NOT:  Load top-level profile of inlinee container: qux
; PRELINK:      Load top-level profile of inlinee container: baz
; PRELINK-NEXT: Load top-level profile of inlinee container: main
; PRELINK-NOT:  Load top-level profile of inlinee container:

; MD5 GUIDs of baz and main.
; PRELINK-MD5:      Load top-level profile of inlinee container: 7546896869197086323
; PRELINK-MD5-NEXT: Load top-level profile of inlinee container: 15822663052811949562
; PRELINK-MD5-NOT:  Load top-level profile of inlinee container:

; NOLOAD-NOT: Load top-level profile of inlinee container

define void @bar() #0 {
  ret void
}

attributes #0 = { "use-sample-profile" }
