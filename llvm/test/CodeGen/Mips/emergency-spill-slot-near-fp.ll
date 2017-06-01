; Check that register scavenging spill slot is close to $fp.
; RUN: llc -march=mipsel -O0 -relocation-model=pic < %s | FileCheck %s

; CHECK: sw ${{.*}}, 8($sp)
; CHECK: lw ${{.*}}, 8($sp)

define i32 @main(i32 signext %argc, i8** %argv) #0 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 4
  %v0 = alloca <16 x i8>, align 16
  %.compoundliteral = alloca <16 x i8>, align 16
  %v1 = alloca <16 x i8>, align 16
  %.compoundliteral1 = alloca <16 x i8>, align 16
  %unused_variable = alloca [16384 x i32], align 4
  %result = alloca <16 x i8>, align 16
  store i32 0, i32* %retval
  store i32 %argc, i32* %argc.addr, align 4
  store i8** %argv, i8*** %argv.addr, align 4
  store <16 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 16>, <16 x i8>* %.compoundliteral
  %0 = load <16 x i8>, <16 x i8>* %.compoundliteral
  store <16 x i8> %0, <16 x i8>* %v0, align 16
  store <16 x i8> zeroinitializer, <16 x i8>* %.compoundliteral1
  %1 = load <16 x i8>, <16 x i8>* %.compoundliteral1
  store <16 x i8> %1, <16 x i8>* %v1, align 16
  %2 = load <16 x i8>, <16 x i8>* %v0, align 16
  %3 = load <16 x i8>, <16 x i8>* %v1, align 16
  %mul = mul <16 x i8> %2, %3
  store <16 x i8> %mul, <16 x i8>* %result, align 16
  ret i32 0
}

attributes #0 = { noinline "no-frame-pointer-elim"="true" }
