;; XFAIL: darwin,sun
;; RUN: echo create > %t.commands
;; RUN: echo s >> %t.commands
;; RUN: echo s >> %t.commands
;; RUN: echo finish >> %t.commands
;; RUN: echo bt >> %t.commands
;; RUN: echo q >> %t.commands
;; RUN: echo y >> %t.commands
;; RUN: llvm-as -f %s -o %t.bc
;; RUN: llvm-db %t.bc < %t.commands | grep 'in main at funccall.c:11:2'


;; Debugger type declarations
%lldb.compile_unit = type { uint, ushort, ushort, sbyte*, sbyte*, sbyte*, {}* }
%lldb.global = type { uint, %lldb.compile_unit*, sbyte*, {}*, sbyte*, bool, bool }
%lldb.local = type { uint, %lldb.global*, sbyte*, sbyte*, bool, bool }


;; Debugger intrinsic declarations...
declare {}* %llvm.dbg.stoppoint({}*, uint, uint, %lldb.compile_unit*)
declare {}* %llvm.dbg.func.start(%lldb.global*)
declare {}* %llvm.dbg.region.start({}*)
declare {}* %llvm.dbg.region.end({}*)
declare {}* %llvm.dbg.declare({}*, ...)

;; Global object anchors
%llvm.dbg.translation_units = linkonce global {} {}
%llvm.dbg.globals = linkonce global {} {}


%.str_1 = internal constant [11 x sbyte] c"funccall.c\00"
%.str_2 = internal constant [12 x sbyte] c"/home/sabre\00"
%.str_3 = internal constant [14 x sbyte] c"llvmgcc 3.4.x\00"

%d.compile_unit = internal constant %lldb.compile_unit {
   uint 17,                                                        ;; DW_TAG_compile_unit
   ushort 0,                                                       ;; LLVM Debug version #
   ushort 1,                                                       ;; Language: DW_LANG_C89
   sbyte* getelementptr ([11 x sbyte]* %.str_1, long 0, long 0),   ;; Source filename
   sbyte* getelementptr ([12 x sbyte]* %.str_2, long 0, long 0),   ;; Working directory
   sbyte* getelementptr ([14 x sbyte]* %.str_3, long 0, long 0),   ;; producer
   {}* %llvm.dbg.translation_units                                 ;; Anchor
}


%.str_5 = internal global [5 x sbyte] c"main\00"
%.str_6 = internal global [4 x sbyte] c"foo\00"
%.str_7 = internal global [2 x sbyte] c"q\00"
%.str_8 = internal global [2 x sbyte] c"t\00"

%d.main = global %lldb.global {
   uint 46,                                                        ;; DW_TAG_subprogram
   %lldb.compile_unit* %d.compile_unit,                            ;; context pointer
   sbyte* getelementptr ([5 x sbyte]* %.str_5, long 0, long 0),    ;; name
   {}* %llvm.dbg.globals,                                          ;; anchor
   sbyte* null,                                                    ;; EVENTUALLY the type
   bool true,                                                      ;; non-static linkage?
  bool false                                                       ;; definition, not declaration
}

%d.foo = global %lldb.global {
   uint 46,                                                        ;; DW_TAG_subprogram
   %lldb.compile_unit* %d.compile_unit,                            ;; context pointer
   sbyte* getelementptr ([4 x sbyte]* %.str_6, long 0, long 0),    ;; name
   {}* %llvm.dbg.globals,                                          ;; anchor
   sbyte* null,                                                    ;; EVENTUALLY the type
   bool true,                                                      ;; non-static linkage
  bool false                                                       ;; definition, not declaration
}

%d.q = internal global %lldb.global {
  uint 52,                                                         ;; DW_TAG_variable
  %lldb.compile_unit* %d.compile_unit,                             ;; context pointer
  sbyte* getelementptr ([2 x sbyte]* %.str_7, long 0, long 0),     ;; name
  {}* %llvm.dbg.globals,                                           ;; anchor
  sbyte* null,                                                     ;; EVENTUALLY the type
  bool false,                                                      ;; static linkage
  bool false                                                       ;; definition, not declaration
}


%d.t = internal global %lldb.local {
  uint 52,                                                         ;; DW_TAG_variable
  %lldb.global* %d.foo,                                            ;; context pointer
  sbyte* getelementptr ([2 x sbyte]* %.str_8, long 0, long 0),     ;; name
  sbyte* null,                                                     ;; EVENTUALLY the type
  bool false,                                                      ;; local variable
  bool false                                                       ;; definition, not declaratation
}



%q = internal global int 0

implementation   ; Functions:

void %foo() {
entry:
	%t = alloca int
	%.1 = call {}* %llvm.dbg.func.start(%lldb.global* %d.foo)
	%.2 = call {}* %llvm.dbg.stoppoint({}* %.1, uint 5, uint 2, %lldb.compile_unit* %d.compile_unit)

        %.3 = call {}*({}*, ...)* %llvm.dbg.declare({}* %.2, %lldb.local* %d.t, int* %t)
	%tmp.0 = load int* %q
	store int %tmp.0, int* %t
	%.4 = call {}* %llvm.dbg.stoppoint({}* %.3, uint 6, uint 2, %lldb.compile_unit* %d.compile_unit)
	%tmp.01 = load int* %t
	%tmp.1 = add int %tmp.01, 1
	store int %tmp.1, int* %q
	%.5 = call {}* %llvm.dbg.stoppoint({}* %.4, uint 7, uint 1, %lldb.compile_unit* %d.compile_unit)
	call {}* %llvm.dbg.region.end({}* %.5)
	ret void
}

int %main() {
entry:
	%.1 = call {}* %llvm.dbg.func.start(%lldb.global* %d.main)
	%result = alloca int
	%.2 = call {}* %llvm.dbg.stoppoint({}* %.1, uint 9, uint 2, %lldb.compile_unit* %d.compile_unit)
	store int 0, int* %q
	%.3 = call {}* %llvm.dbg.stoppoint({}* %.2, uint 10, uint 2, %lldb.compile_unit* %d.compile_unit)
	call void %foo()
	%.4 = call {}* %llvm.dbg.stoppoint({}* %.3, uint 11, uint 2, %lldb.compile_unit* %d.compile_unit)
	%tmp.2 = load int* %q
	%tmp.3 = sub int %tmp.2, 1
	store int %tmp.3, int* %q
	%.5 = call {}* %llvm.dbg.stoppoint({}* %.4, uint 13, uint 2, %lldb.compile_unit* %d.compile_unit)
	%tmp.4 = load int* %q
	store int %tmp.4, int* %result
	%tmp.5 = load int* %result
	%.6 = call {}* %llvm.dbg.stoppoint({}* %.5, uint 14, uint 1, %lldb.compile_unit* %d.compile_unit)
	call {}* %llvm.dbg.region.end({}* %.6)
	ret int %tmp.5
}
