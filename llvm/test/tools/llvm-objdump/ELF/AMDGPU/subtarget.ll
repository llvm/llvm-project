define amdgpu_kernel void @test_kernel() {
  ret void
}

; Test subtarget detection. Disassembly is only supported for GFX8 and beyond.
;
; ----------------------------------GFX13--------------------------------------
;
; RUN: llc -mtriple=amdgpu13.10-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt
;
; RUN: llc -mtriple=amdgpu13-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu -mllvm --amdhsa-code-object-version=6 %t.o > %t-specify.txt
; RUN: llvm-objdump -D -mllvm --amdhsa-code-object-version=6 %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt
;
; ----------------------------------GFX12--------------------------------------
;
; RUN: llc -mtriple=amdgpu12-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu -mllvm --amdhsa-code-object-version=6 %t.o > %t-specify.txt
; RUN: llvm-objdump -D -mllvm --amdhsa-code-object-version=6 %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt
;
; RUN: llc -mtriple=amdgpu12.01-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt
;
; RUN: llc -mtriple=amdgpu12.00-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu12.5-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu -mllvm --amdhsa-code-object-version=6 %t.o > %t-specify.txt
; RUN: llvm-objdump -D -mllvm --amdhsa-code-object-version=6 %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu12.50s-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu12.51-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; ----------------------------------GFX11--------------------------------------
;
; RUN: llc -mtriple=amdgpu11.7-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu -mllvm --amdhsa-code-object-version=6 %t.o > %t-specify.txt
; RUN: llvm-objdump -D -mllvm --amdhsa-code-object-version=6 %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu11-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu -mllvm --amdhsa-code-object-version=6 %t.o > %t-specify.txt
; RUN: llvm-objdump -D -mllvm --amdhsa-code-object-version=6 %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu11.72-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu11.71-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu11.70-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu11.54-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu11.53-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu11.52-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu11.51-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu11.50-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu11.03-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu11.02-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu11.01-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; ----------------------------------GFX10--------------------------------------
; RUN: llc -mtriple=amdgpu10.3-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu -mllvm --amdhsa-code-object-version=6 %t.o > %t-specify.txt
; RUN: llvm-objdump -D  -mllvm --amdhsa-code-object-version=6 %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu10.36-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu10.35-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu10.34-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu10.33-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu10.32-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu10.31-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu10.30-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu10.1-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu -mllvm --amdhsa-code-object-version=6 %t.o > %t-specify.txt
; RUN: llvm-objdump -D  -mllvm --amdhsa-code-object-version=6 %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu10.13-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu10.12-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu10.11-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu10.10-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt


; ----------------------------------GFX9---------------------------------------
;
; RUN: llc -mtriple=amdgpu9.4-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu -mllvm --amdhsa-code-object-version=6 %t.o > %t-specify.txt
; RUN: llvm-objdump -D  -mllvm --amdhsa-code-object-version=6 %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu9-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu -mllvm --amdhsa-code-object-version=6 %t.o > %t-specify.txt
; RUN: llvm-objdump -D  -mllvm --amdhsa-code-object-version=6 %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu9.0c-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu9.09-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu9.08-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu9.06-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu9.04-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu9.02-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt


; ----------------------------------GFX8---------------------------------------
;
; RUN: llc -mtriple=amdgpu8.10-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu8.03-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu8.02-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu8.01-amd-amdhsa -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu %t.o > %t-specify.txt
; RUN: llvm-objdump -D %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; ------------------------Generic triple + -mcpu-------------------------------
;
; A generic-family triple may be refined by -mcpu naming a covered subtarget.
;
; RUN: llc -mtriple=amdgpu12-amd-amdhsa --amdhsa-code-object-version=6 -mcpu=gfx1200 -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu -mllvm --amdhsa-code-object-version=6 %t.o > %t-specify.txt
; RUN: llvm-objdump -D -mllvm --amdhsa-code-object-version=6 %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu10.3-amd-amdhsa --amdhsa-code-object-version=6 -mcpu=gfx1030 -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu -mllvm --amdhsa-code-object-version=6 %t.o > %t-specify.txt
; RUN: llvm-objdump -D -mllvm --amdhsa-code-object-version=6 %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt

; RUN: llc -mtriple=amdgpu9-amd-amdhsa --amdhsa-code-object-version=6 -mcpu=gfx900 -filetype=obj -O0 -o %t.o %s
; RUN: llvm-objdump -D --arch-name=amdgpu -mllvm --amdhsa-code-object-version=6 %t.o > %t-specify.txt
; RUN: llvm-objdump -D -mllvm --amdhsa-code-object-version=6 %t.o > %t-detect.txt
; RUN: diff %t-specify.txt %t-detect.txt
