; Tests that data and relro are correctly placed in sections
; specified by "#pragma clang section"
; RUN: llc -filetype=obj -mtriple x86_64-unknown-linux %s -o - | llvm-readobj -S -t - | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

@funcs_relro = hidden constant [2 x i32 ()*] [i32 ()* bitcast (i32 (...)* @func1 to i32 ()*), i32 ()* bitcast (i32 (...)* @func2 to i32 ()*)], align 16 #0
@var_data = hidden global i32 33, align 4 #0

declare i32 @func1(...)
declare i32 @func2(...)

; Function Attrs: noinline nounwind optnone sspstrong uwtable
define hidden i32 @foo(i32 %i) {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %idxprom = sext i32 %0 to i64
  %arrayidx = getelementptr inbounds [2 x i32 ()*], [2 x i32 ()*]* @funcs_relro, i64 0, i64 %idxprom
  %1 = load i32 ()*, i32 ()** %arrayidx, align 8
  %call = call i32 %1()
  %2 = load i32, i32* @var_data, align 4
  %add = add nsw i32 %call, %2
  ret i32 %add
}

attributes #0 = { "data-section"=".my_data" "relro-section"=".my_relro" "rodata-section"=".my_rodata" }

; CHECK:  Section {
; CHECK:    Index:
; CHECK:    Name: .my_rodata
; CHECK:    Type: SHT_PROGBITS (0x1)
; CHECK:    Flags [ (0x2)
; CHECK:      SHF_ALLOC (0x2)
; CHECK:    ]
; CHECK:    Size: 16
; CHECK:  }
; CHECK:  Section {
; CHECK:    Index:
; CHECK:    Name: .my_data
; CHECK:    Type: SHT_PROGBITS (0x1)
; CHECK:    Flags [ (0x3)
; CHECK:      SHF_ALLOC (0x2)
; CHECK:      SHF_WRITE (0x1)
; CHECK:    ]
; CHECK:    Size: 4
; CHECK:  }
; CHECK:   Symbol {
; CHECK:    Name: funcs_relro
; CHECK:    Value: 0x0
; CHECK:    Size: 16
; CHECK:    Binding: Global (0x1)
; CHECK:    Type: Object (0x1)
; CHECK:    Section: .my_rodata
; CHECK:  }
; CHECK:  Symbol {
; CHECK:    Name: var_data
; CHECK:    Value: 0x0
; CHECK:    Size: 4
; CHECK:    Binding: Global (0x1)
; CHECK:    Type: Object (0x1)
; CHECK:    Section: .my_data
; CHECK:  }
