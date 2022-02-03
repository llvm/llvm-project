;; Test valid and invalid -binutils-version values.
; REQUIRES: default_target
; RUN: llc %s -filetype=null -binutils-version=none
; RUN: llc %s -filetype=null -binutils-version=2
; RUN: llc %s -filetype=null -binutils-version=2.35

;; Disallow -binutils-version=0 because we use $major==0 to indicate the MC
;; default.
; RUN: not llc %s -filetype=null -binutils-version=0 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc %s -filetype=null -binutils-version=nan 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc %s -filetype=null -binutils-version=2. 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc %s -filetype=null -binutils-version=3.-14 2>&1 | FileCheck %s --check-prefix=ERR

; ERR: error: invalid -binutils-version, accepting 'none' or major.minor
