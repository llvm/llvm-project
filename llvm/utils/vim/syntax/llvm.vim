" Vim syntax file
" Language:   llvm
" Maintainer: The LLVM team, http://llvm.org/
" Version:      $Revision$

if version < 600
  syntax clear
elseif exists("b:current_syntax")
  finish
endif

syn case match

" Types.
" Types also include struct, array, vector, etc. but these don't
" benefit as much from having dedicated highlighting rules.
syn keyword llvmType void half bfloat float double x86_fp80 fp128 ppc_fp128
syn keyword llvmType label metadata x86_mmx x86_amx
syn keyword llvmType type label opaque token
syn match   llvmType /\<i\d\+\>/

" Instructions.
" The true and false tokens can be used for comparison opcodes, but it's
" much more common for these tokens to be used for boolean constants.
syn keyword llvmStatement add addrspacecast alloca and arcp ashr atomicrmw
syn keyword llvmStatement bitcast br catchpad catchswitch catchret call callbr
syn keyword llvmStatement cleanuppad cleanupret cmpxchg eq exact extractelement
syn keyword llvmStatement extractvalue fadd fast fcmp fdiv fence fmul fpext
syn keyword llvmStatement fptosi fptoui fptrunc free frem fsub fneg getelementptr
syn keyword llvmStatement icmp inbounds indirectbr insertelement insertvalue
syn keyword llvmStatement inttoptr invoke landingpad load lshr malloc max min
syn keyword llvmStatement mul nand ne ninf nnan nsw nsz nuw oeq oge ogt ole
syn keyword llvmStatement olt one or ord phi ptrtoint resume ret sdiv select
syn keyword llvmStatement sext sge sgt shl shufflevector sitofp sle slt srem
syn keyword llvmStatement store sub switch trunc udiv ueq uge ugt uitofp ule ult
syn keyword llvmStatement umax umin une uno unreachable unwind urem va_arg
syn keyword llvmStatement xchg xor zext

" Keywords.
syn keyword llvmKeyword
      \ acq_rel
      \ acquire
      \ addrspace
      \ alias
      \ align
      \ alignstack
      \ alwaysinline
      \ appending
      \ argmemonly
      \ arm_aapcscc
      \ arm_aapcs_vfpcc
      \ arm_apcscc
      \ asm
      \ atomic
      \ available_externally
      \ blockaddress
      \ builtin
      \ byref
      \ byval
      \ c
      \ catch
      \ caller
      \ cc
      \ ccc
      \ cleanup
      \ coldcc
      \ comdat
      \ common
      \ constant
      \ datalayout
      \ declare
      \ default
      \ define
      \ deplibs
      \ dereferenceable
      \ distinct
      \ dllexport
      \ dllimport
      \ dso_local
      \ dso_preemptable
      \ except
      \ external
      \ externally_initialized
      \ extern_weak
      \ fastcc
      \ tailcc
      \ filter
      \ from
      \ gc
      \ global
      \ hhvmcc
      \ hhvm_ccc
      \ hidden
      \ immarg
      \ initialexec
      \ inlinehint
      \ inreg
      \ inteldialect
      \ intel_ocl_bicc
      \ internal
      \ linkonce
      \ linkonce_odr
      \ localdynamic
      \ localexec
      \ local_unnamed_addr
      \ minsize
      \ module
      \ monotonic
      \ msp430_intrcc
      \ mustprogress
      \ musttail
      \ naked
      \ nest
      \ noalias
      \ nobuiltin
      \ nocapture
      \ noimplicitfloat
      \ noinline
      \ nonlazybind
      \ nonnull
      \ norecurse
      \ noredzone
      \ noreturn
      \ noundef
      \ nounwind
      \ optnone
      \ optsize
      \ personality
      \ private
      \ protected
      \ ptx_device
      \ ptx_kernel
      \ readnone
      \ readonly
      \ release
      \ returned
      \ returns_twice
      \ sanitize_address
      \ sanitize_memory
      \ sanitize_thread
      \ section
      \ seq_cst
      \ sideeffect
      \ signext
      \ syncscope
      \ source_filename
      \ speculatable
      \ spir_func
      \ spir_kernel
      \ sret
      \ ssp
      \ sspreq
      \ sspstrong
      \ strictfp
      \ swiftcc
      \ swiftself
      \ tail
      \ target
      \ thread_local
      \ to
      \ triple
      \ unnamed_addr
      \ unordered
      \ uselistorder
      \ uselistorder_bb
      \ uwtable
      \ volatile
      \ weak
      \ weak_odr
      \ within
      \ writeonly
      \ x86_64_sysvcc
      \ win64cc
      \ x86_fastcallcc
      \ x86_stdcallcc
      \ x86_thiscallcc
      \ zeroext

" Obsolete keywords.
syn keyword llvmError  getresult begin end

" Misc syntax.
syn match   llvmNoName /[%@!]\d\+\>/
syn match   llvmNumber /-\?\<\d\+\>/
syn match   llvmFloat  /-\?\<\d\+\.\d*\(e[+-]\d\+\)\?\>/
syn match   llvmFloat  /\<0x\x\+\>/
syn keyword llvmBoolean true false
syn keyword llvmConstant zeroinitializer undef null none
syn match   llvmComment /;.*$/
syn region  llvmString start=/"/ skip=/\\"/ end=/"/
syn match   llvmLabel /[-a-zA-Z$._][-a-zA-Z$._0-9]*:/
syn match   llvmIdentifier /[%@][-a-zA-Z$._][-a-zA-Z$._0-9]*/

" Named metadata and specialized metadata keywords.
syn match   llvmIdentifier /![-a-zA-Z$._][-a-zA-Z$._0-9]*\ze\s*$/
syn match   llvmIdentifier /![-a-zA-Z$._][-a-zA-Z$._0-9]*\ze\s*[=!]/
syn match   llvmType /!\zs\a\+\ze\s*(/
syn match   llvmConstant /\<DW_TAG_[a-z_]\+\>/
syn match   llvmConstant /\<DW_ATE_[a-zA-Z_]\+\>/
syn match   llvmConstant /\<DW_OP_[a-zA-Z0-9_]\+\>/
syn match   llvmConstant /\<DW_LANG_[a-zA-Z0-9_]\+\>/
syn match   llvmConstant /\<DW_VIRTUALITY_[a-z_]\+\>/
syn match   llvmConstant /\<DIFlag[A-Za-z]\+\>/

" Syntax-highlight lit test commands and bug numbers.
syn match  llvmSpecialComment /;\s*PR\d*\s*$/
syn match  llvmSpecialComment /;\s*REQUIRES:.*$/
syn match  llvmSpecialComment /;\s*RUN:.*$/
syn match  llvmSpecialComment /;\s*ALLOW_RETRIES:.*$/
syn match  llvmSpecialComment /;\s*CHECK:.*$/
syn match  llvmSpecialComment "\v;\s*CHECK-(NEXT|NOT|DAG|SAME|LABEL):.*$"
syn match  llvmSpecialComment /;\s*XFAIL:.*$/

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
  HiLink llvmNoName Identifier
  HiLink llvmConstant Constant
  HiLink llvmSpecialComment SpecialComment
  HiLink llvmError Error
  HiLink llvmIdentifier Identifier

  delcommand HiLink
endif

let b:current_syntax = "llvm"
