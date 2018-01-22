; Check msa warnings.
; RUN: llc -march=mips -mattr=+mips32r2 -mattr=+msa -mattr=+fp64 < %s 2>&1 | \
; RUN:   FileCheck %s -check-prefix=MSA_32
; RUN: llc -march=mips64 -mattr=+mips64r2 -mattr=+msa < %s 2>&1 | \
; RUN:   FileCheck %s  -check-prefix=MSA_64
; RUN: llc -march=mips -mattr=+mips32r5 -mattr=+msa -mattr=+fp64 < %s 2>&1 | \
; RUN:   FileCheck %s -check-prefix=MSA_32_NO_WARNING
; RUN: llc -march=mips64 -mattr=+mips64r5 -mattr=+msa < %s 2>&1 | \
; RUN:   FileCheck %s  -check-prefix=MSA_64_NO_WARNING

; Check dspr2 warnings.
; RUN: llc -march=mips -mattr=+mips32 -mattr=+dspr2 < %s 2>&1 | \
; RUN:   FileCheck %s -check-prefix=DSPR2_32
; RUN: llc -march=mips64 -mattr=+mips64 -mattr=+dspr2 < %s 2>&1 | \
; RUN:   FileCheck %s -check-prefix=DSPR2_64
; RUN: llc -march=mips64 -mattr=+mips64r3 -mattr=+dspr2 < %s  2>&1 | \
; RUN:   FileCheck %s -check-prefix=DSPR2_64_NO_WARNING
; RUN: llc -march=mips -mattr=+mips32r2 -mattr=+dspr2 < %s 2>&1 | \
; RUN:   FileCheck %s  -check-prefix=DSPR2_32_NO_WARNING

; Check dsp warnings.
; RUN: llc -march=mips -mattr=+mips32 -mattr=+dsp < %s 2>&1 | \
; RUN:   FileCheck %s -check-prefix=DSP_32
; RUN: llc -march=mips64 -mattr=+mips64 -mattr=+dsp < %s 2>&1 | \
; RUN:   FileCheck %s  -check-prefix=DSP_64
; RUN: llc -march=mips -mattr=+mips32r5 -mattr=+dsp < %s 2>&1 | \
; RUN:   FileCheck %s -check-prefix=DSP_32_NO_WARNING
; RUN: llc -march=mips64 -mattr=+mips64r2 -mattr=+dsp < %s 2>&1 | \
; RUN:   FileCheck %s -check-prefix=DSP_64_NO_WARNING

; MSA_32: warning: the 'msa' ASE requires MIPS32 revision 5 or greater
; MSA_64: warning: the 'msa' ASE requires MIPS64 revision 5 or greater
; MSA_32_NO_WARNING-NOT: warning: the 'msa' ASE requires MIPS32 revision 5 or greater
; MSA_64_NO_WARNING-NOT: warning: the 'msa' ASE requires MIPS64 revision 5 or greater

; DSPR2_32: warning: the 'dspr2' ASE requires MIPS32 revision 2 or greater
; DSPR2_64: warning: the 'dspr2' ASE requires MIPS64 revision 2 or greater
; DSPR2_32_NO_WARNING-NOT: warning: the 'dspr2' ASE requires MIPS32 revision 2 or greater
; DSPR2_64_NO_WARNING-NOT: warning: the 'dspr2' ASE requires MIPS64 revision 2 or greater

; DSP_32: warning: the 'dsp' ASE requires MIPS32 revision 2 or greater
; DSP_64: warning: the 'dsp' ASE requires MIPS64 revision 2 or greater
; DSP_32_NO_WARNING-NOT: warning: the 'dsp' ASE requires MIPS32 revision 2 or greater
; DSP_64_NO_WARNING-NOT: warning: the 'dsp' ASE requires MIPS64 revision 2 or greater
