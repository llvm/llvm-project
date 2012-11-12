;; LLVM coding style guidelines in emacs
;; Maintainer: LLVM Team, http://llvm.org/
;; Modified:   2009-07-28

;; Max 80 cols per line, indent by two spaces, no tabs.
;; Apparently, this does not affect tabs in Makefiles.
(custom-set-variables
  '(fill-column 80)
  '(c++-indent-level 2)
  '(c-basic-offset 2)
  '(indent-tabs-mode nil))


;; Alternative to setting the global style.  Only files with "llvm" in
;; their names will automatically set to the llvm.org coding style.
(c-add-style "llvm.org"
             '((fill-column . 80)
	       (c++-indent-level . 2)
	       (c-basic-offset . 2)
	       (indent-tabs-mode . nil)
               (c-offsets-alist . ((innamespace 0)))))

(add-hook 'c-mode-hook
	  (function
	   (lambda nil 
	     (if (string-match "llvm" buffer-file-name)
		 (progn
		   (c-set-style "llvm.org")
		   )
	       ))))

(add-hook 'c++-mode-hook
	  (function
	   (lambda nil 
	     (if (string-match "llvm" buffer-file-name)
		 (progn
		   (c-set-style "llvm.org")
		   )
	       ))))
