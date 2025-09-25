; RUN: llc < %s -mtriple=sparc -mattr=-v9 | FileCheck %s -check-prefix=V8
; RUN: llc < %s -mtriple=sparc -mattr=+v9,+popc | FileCheck %s -check-prefix=V9
; RUN: llc < %s -mtriple=sparc -mcpu=v9 | FileCheck %s -check-prefix=V8
; RUN: llc < %s -mtriple=sparc -mcpu=ultrasparc | FileCheck %s -check-prefix=V8
; RUN: llc < %s -mtriple=sparc -mcpu=ultrasparc3 | FileCheck %s -check-prefix=V8
; RUN: llc < %s -mtriple=sparc -mcpu=niagara | FileCheck %s -check-prefix=V8
; RUN: llc < %s -mtriple=sparc -mcpu=niagara2 | FileCheck %s -check-prefix=V9
; RUN: llc < %s -mtriple=sparc -mcpu=niagara3 | FileCheck %s -check-prefix=V9
; RUN: llc < %s -mtriple=sparc -mcpu=niagara4 | FileCheck %s -check-prefix=V9
; RUN: llc < %s -mtriple=sparcv9 -mattr=+popc | FileCheck %s -check-prefix=SPARC64

declare i32 @llvm.ctpop.i32(i32)

; V8-LABEL: test
; V8-NOT: popc

; V9-LABEL: test
; V9:       srl %o0, 0, %o0
; V9-NEXT:  retl
; V9-NEXT:  popc %o0, %o0

; SPARC64-LABEL: test
; SPARC64:       srl %o0, 0, %o0
; SPARC64:       retl
; SPARC64:       popc %o0, %o0

define i32 @test(i32 %X) {
        %Y = call i32 @llvm.ctpop.i32( i32 %X )         ; <i32> [#uses=1]
        ret i32 %Y
}

