;; LLVM coding style guidelines in emacs
;; Maintainer: LLVM Team, http://llvm.org/
;; Modified:   2005-04-24

;; Max 80 cols per line, indent by two spaces, no tabs.
;; Apparently, this does not affect tabs in Makefiles.
(custom-set-variables
  '(fill-column 80)
  '(c++-indent-level 2)
  '(c-basic-offset 2)
  '(indent-tabs-mode nil))

