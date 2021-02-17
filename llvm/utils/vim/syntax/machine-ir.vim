" Vim syntax file
" Language:   mir
" Maintainer: The LLVM team, http://llvm.org/
" Version:      $Revision$

" FIXME: MIR doesn't actually match LLVM IR. Reimplement this.
" See the MIR LangRef: https://llvm.org/docs/MIRLangRef.html
unlet b:current_syntax  " Unlet so that the LLVM syntax will load
runtime! syntax/llvm.vim
unlet b:current_syntax
let b:current_syntax = "mir"
