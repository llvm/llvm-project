; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 -arm-global-merge -global-merge-group-by-use=false | FileCheck %s
; CHECK: .zerofill __DATA,__bss,l__MergedGlobals,16,2

@prev = external global [0 x i16]
@max_lazy_match = internal unnamed_addr global i32 0, align 4
@read_buf = external global i32 (i8*, i32)*
@window = external global [0 x i8]
@lookahead = internal unnamed_addr global i32 0, align 4
@eofile.b = internal unnamed_addr global i32 0
@ins_h = internal unnamed_addr global i32 0, align 4
