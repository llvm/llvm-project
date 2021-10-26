" Vim syntax file
" Language:   mlir
" Maintainer: The MLIR team, http://github.com/tensorflow/mlir/
" Version:      $Revision$
" Some parts adapted from the LLVM vim syntax file.

if version < 600
  syntax clear
elseif exists("b:current_syntax")
  finish
endif

syn case match

" Types.
"
syn keyword mlirType index f16 f32 f64
" Signless integer types.
syn match mlirType /\<i\d\+\>/
" Unsigned integer types.
syn match mlirType /\<ui\d\+\>/
" Signed integer types.
syn match mlirType /\<si\d\+\>/

" Elemental types inside memref, tensor, or vector types.
syn match mlirType /x\s*\zs\(f16\|f32\|f64\|i\d\+\|ui\d\+\|si\d\+\)/

" Shaped types.
syn match mlirType /\<memref\ze\s*<.*>/
syn match mlirType /\<tensor\ze\s*<.*>/
syn match mlirType /\<vector\ze\s*<.*>/

" vector types inside memref or tensor.
syn match mlirType /x\s*\zsvector/

" Operations.
" Standard dialect ops.
" TODO: this list is not exhaustive.
syn keyword mlirOps alloc alloca addf addi and call call_indirect cmpf cmpi
syn keyword mlirOps constant dealloc divf dma_start dma_wait dim exp
syn keyword mlirOps getTensor index_cast load log memref_cast
syn keyword mlirOps memref_shape_cast mulf muli negf powf prefetch rsqrt sitofp
syn keyword mlirOps splat store select sqrt subf subi subview tanh
syn keyword mlirOps view

" Math ops.
syn match mlirOps /\<math\.erf\>/

" Affine ops.
syn match mlirOps /\<affine\.apply\>/
syn match mlirOps /\<affine\.dma_start\>/
syn match mlirOps /\<affine\.dma_wait\>/
syn match mlirOps /\<affine\.for\>/
syn match mlirOps /\<affine\.if\>/
syn match mlirOps /\<affine\.load\>/
syn match mlirOps /\<affine\.parallel\>/
syn match mlirOps /\<affine\.prefetch\>/
syn match mlirOps /\<affine\.store\>/
syn match mlirOps /\<scf\.execute_region\>/
syn match mlirOps /\<scf\.for\>/
syn match mlirOps /\<scf\.if\>/
syn match mlirOps /\<scf\.yield\>/

" TODO: dialect name prefixed ops (llvm or std).

" Keywords.
syn keyword mlirKeyword
      \ affine_map
      \ affine_set
      \ dense
      \ else
      \ func
      \ module
      \ return
      \ step
      \ to

" Misc syntax.

syn match   mlirNumber /-\?\<\d\+\>/
" Match numbers even in shaped types.
syn match   mlirNumber /-\?\<\d\+\ze\s*x/
syn match   mlirNumber /x\s*\zs-\?\d\+\ze\s*x/

syn match   mlirFloat  /-\?\<\d\+\.\d*\(e[+-]\d\+\)\?\>/
syn match   mlirFloat  /\<0x\x\+\>/
syn keyword mlirBoolean true false
syn match   mlirComment /\/\/.*$/
syn region  mlirString start=/"/ skip=/\\"/ end=/"/
syn match   mlirLabel /[-a-zA-Z$._][-a-zA-Z$._0-9]*:/
syn match   mlirIdentifier /[%@][a-zA-Z$._-][a-zA-Z0-9$._-]*/
syn match   mlirIdentifier /[%@!]\d\+\>/
syn match mlirMapSetOutline "#.*$"

" Syntax-highlight lit test commands and bug numbers.
syn match  mlirSpecialComment /\/\/\s*RUN:.*$/
syn match  mlirSpecialComment /\/\/\s*CHECK:.*$/
syn match  mlirSpecialComment "\v\/\/\s*CHECK-(NEXT|NOT|DAG|SAME|LABEL):.*$"
syn match  mlirSpecialComment /\/\/\s*expected-error.*$/
syn match  mlirSpecialComment /\/\/\s*expected-remark.*$/
syn match  mlirSpecialComment /;\s*XFAIL:.*$/
syn match  mlirSpecialComment /\/\/\s*PR\d*\s*$/
syn match  mlirSpecialComment /\/\/\s*REQUIRES:.*$/

if version >= 508 || !exists("did_c_syn_inits")
  if version < 508
    let did_c_syn_inits = 1
    command -nargs=+ HiLink hi link <args>
  else
    command -nargs=+ HiLink hi def link <args>
  endif

  HiLink mlirType Type
  HiLink mlirOps Statement
  HiLink mlirMapSetOutline PreProc
  HiLink mlirNumber Number
  HiLink mlirComment Comment
  HiLink mlirString String
  HiLink mlirLabel Label
  HiLink mlirKeyword Keyword
  HiLink mlirBoolean Boolean
  HiLink mlirFloat Float
  HiLink mlirConstant Constant
  HiLink mlirSpecialComment SpecialComment
  HiLink mlirIdentifier Identifier

  delcommand HiLink
endif

let b:current_syntax = "mlir"
