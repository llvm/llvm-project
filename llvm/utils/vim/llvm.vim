" Vim syntax file
" Language:   llvm
" Maintainer: The LLVM team, http://llvm.org/
" Updated:    2003-06-02

if version < 600
  syntax clear
elseif exists("b:current_syntax")
  finish
endif

syn case match

" Types.
" Types also include struct, array, vector, etc. but these don't
" benefit as much from having dedicated highlighting rules.
syn keyword llvmType void float double
syn keyword llvmType x86_fp80 fp128 ppc_fp128
syn keyword llvmType type label opaque
syn match   llvmType /\<i\d\+\>/

" Instructions.
" The true and false tokens can be used for comparison opcodes, but it's
" much more common for these tokens to be used for boolean constants.
syn keyword llvmStatement add sub mul sdiv udiv fdiv srem urem frem
syn keyword llvmStatement and or xor
syn keyword llvmStatement icmp fcmp
syn keyword llvmStatement eq ne ugt uge ult ule sgt sge slt sle
syn keyword llvmStatement oeq ogt oge olt ole one ord ueq ugt uge
syn keyword llvmStatement ult ule une uno
syn keyword llvmStatement phi call select shl lshr ashr va_arg
syn keyword llvmStatement trunc zext sext
syn keyword llvmStatement fptrunc fpext fptoui fptosi uitofp sitofp
syn keyword llvmStatement ptrtoint inttoptr bitcast
syn keyword llvmStatement ret br switch invoke unwind unreachable
syn keyword llvmStatement malloc alloca free load store getelementptr
syn keyword llvmStatement extractelement insertelement shufflevector
syn keyword llvmStatement extractvalue insertvalue

" Keywords.
syn keyword llvmKeyword define declare global constant
syn keyword llvmKeyword internal external
syn keyword llvmKeyword linkonce weak appending common extern_weak
syn keyword llvmKeyword thread_local dllimport dllexport
syn keyword llvmKeyword hidden protected default
syn keyword llvmKeyword except deplibs
syn keyword llvmKeyword volatile fastcc coldcc cc ccc
syn keyword llvmKeyword x86_stdcallcc x86_fastcallcc
syn keyword llvmKeyword signext zeroext inreg sret nounwind noreturn
syn keyword llvmKeyword nocapture byval nest readnone readonly
syn keyword llvmKeyword noinline alwaysinline optsize ssp sspreq
syn keyword llvmKeyword module asm align tail to
syn keyword llvmKeyword addrspace section alias sideeffect c gc
syn keyword llvmKeyword target datalayout triple

" Obsolete keywords.
syn keyword llvmError  uninitialized implementation
syn keyword llvmError  getresult big little endian begin end

" Misc syntax.
syn match   llvmIgnore /[%@]\d\+\>/
syn match   llvmNumber /-\?\<\d\+\>/
syn match   llvmFloat  /-\?\<\d\+\.\d*\(e[+-]\d\+\)\?\>/
syn match   llvmFloat  /\<0x\x\+\>/
syn keyword llvmBoolean true false
syn keyword llvmConstant zeroinitializer undef null
syn match   llvmComment /;.*$/
syn region  llvmString start=/"/ skip=/\\"/ end=/"/
syn match   llvmLabel /[\-a-zA-Z\$._0-9]*:/

" Syntax-highlight dejagnu test commands.
syn match  llvmSpecialComment /;\s*RUN:.*$/
syn match  llvmSpecialComment /;\s*PR\d*\s*$/
syn match  llvmSpecialComment /;\s*END\.\s*$/
syn match  llvmSpecialComment /;\s*XFAIL:.*$/
syn match  llvmSpecialComment /;\s*XTARGET:.*$/

if version >= 508 || !exists("did_c_syn_inits")
  if version < 508
    let did_c_syn_inits = 1
    command -nargs=+ HiLink hi link <args>
  else
    command -nargs=+ HiLink hi def link <args>
  endif

  HiLink llvmType Type
  HiLink llvmStatement Statement
  HiLink llvmNumber Number
  HiLink llvmComment Comment
  HiLink llvmString String
  HiLink llvmLabel Label
  HiLink llvmKeyword Keyword
  HiLink llvmBoolean Boolean
  HiLink llvmFloat Float
  HiLink llvmIgnore Ignore
  HiLink llvmConstant Constant
  HiLink llvmSpecialComment SpecialComment
  HiLink llvmError Error

  delcommand HiLink
endif

let b:current_syntax = "llvm"
