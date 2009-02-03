;; Maintainer:  The LLVM team, http://llvm.org/
;; Description: Major mode for the LLVM assembler language.
;; Updated:     2007-09-19

;; Create mode-specific tables.
(defvar llvm-mode-syntax-table nil
  "Syntax table used while in LLVM mode.")

(defvar llvm-font-lock-keywords
  (list
   ;; Comments
   '(";.*" . font-lock-comment-face)
   ;; Variables
   '("%[-a-zA-Z$\._][-a-zA-Z$\._0-9]*" . font-lock-variable-name-face)
   ;; Labels
   '("[-a-zA-Z$\._0-9]+:" . font-lock-variable-name-face)
   ;; Strings
   '("\"[^\"]+\"" . font-lock-string-face)
   ;; Unnamed variable slots
   '("%[-]?[0-9]+" . font-lock-variable-name-face)
   ;; Types
   '("\\bvoid\\b\\|\\bi[0-9]+\\b\\|\\float\\b\\|\\bdouble\\b\\|\\btype\\b\\|\\blabel\\b\\|\\bopaque\\b" . font-lock-type-face)
   ;; Integer literals
   '("\\b[-]?[0-9]+\\b" . font-lock-preprocessor-face)
   ;; Floating point constants
   '("\\b[-+]?[0-9]+\.[0-9]*\([eE][-+]?[0-9]+\)?\\b" . font-lock-preprocessor-face)
   ;; Hex constants
   '("\\b0x[0-9A-Fa-f]+\\b" . font-lock-preprocessor-face)
   ;; Keywords
   '("\\bbegin\\b\\|\\bend\\b\\|\\btrue\\b\\|\\bfalse\\b\\|\\bzeroinitializer\\b\\|\\bdeclare\\b\\|\\bdefine\\b\\|\\bglobal\\b\\|\\bconstant\\b\\|\\bconst\\b\\|\\binternal\\b\\|\\blinkonce\\b\\|\\bweak\\b\\|\\bappending\\b\\|\\buninitialized\\b\\|\\bimplementation\\b\\|\\b\\.\\.\\.\\b\\|\\bnull\\b\\|\\bundef\\b\\|\\bto\\b\\|\\bexcept\\b\\|\\bnot\\b\\|\\btarget\\b\\|\\bendian\\b\\|\\blittle\\b\\|\\bbig\\b\\|\\bpointersize\\b\\|\\bdeplibs\\b\\|\\bvolatile\\b\\|\\bfastcc\\b\\|\\bcoldcc\\b\\|\\bcc\\b" . font-lock-keyword-face)
   ;; Arithmetic and Logical Operators
   '("\\badd\\b\\|\\bsub\\b\\|\\bmul\\b\\|\\bdiv\\b\\|\\brem\\b\\|\\band\\b\\|\\bor\\b\\|\\bxor\\b\\|\\bset\\(ne\\b\\|\\beq\\b\\|\\blt\\b\\|\\bgt\\b\\|\\ble\\b\\|\\bge\\b\\)" . font-lock-keyword-face)
   ;; Special instructions
   '("\\bphi\\b\\|\\btail\\b\\|\\bcall\\b\\|\\bcast\\b\\|\\bselect\\b\\|\\bto\\b\\|\\bshl\\b\\|\\bshr\\b\\|\\bvaarg\\b\\|\\bvanext\\b" . font-lock-keyword-face)
   ;; Control instructions
   '("\\bret\\b\\|\\bbr\\b\\|\\bswitch\\b\\|\\binvoke\\b\\|\\bunwind\\b\\|\\bunreachable\\b" . font-lock-keyword-face)
   ;; Memory operators
   '("\\bmalloc\\b\\|\\balloca\\b\\|\\bfree\\b\\|\\bload\\b\\|\\bstore\\b\\|\\bgetelementptr\\b" . font-lock-keyword-face)
   )
  "Syntax highlighting for LLVM"
  )

;; ---------------------- Syntax table ---------------------------
;; Shamelessly ripped from jasmin.el
;; URL: http://www.neilvandyke.org/jasmin-emacs/jasmin.el.html

(if (not llvm-mode-syntax-table)
    (progn
      (setq llvm-mode-syntax-table (make-syntax-table))
      (mapcar (function (lambda (n)
                          (modify-syntax-entry (aref n 0)
                                               (aref n 1)
                                               llvm-mode-syntax-table)))
              '(
                ;; whitespace (` ')
                [?\^m " "]
                [?\f  " "]
                [?\n  " "]
                [?\t  " "]
                [?\   " "]
                ;; word constituents (`w')
                ;;[?<  "w"]
                ;;[?>  "w"]
                [?\%  "w"]
                ;;[?_  "w  "]
                ;; comments
                [?\;  "< "]
                [?\n  "> "]
                ;;[?\r  "> "]
                ;;[?\^m "> "]
                ;; symbol constituents (`_')
                ;; punctuation (`.')
                ;; open paren (`(')
                [?\( "("]
                [?\[ "("]
                [?\{ "("]
                ;; close paren (`)')
                [?\) ")"]
                [?\] ")"]
                [?\} ")"]
                ;; string quote ('"')
                [?\" "\""]
                ))))

;; --------------------- Abbrev table -----------------------------

(defvar llvm-mode-abbrev-table nil
  "Abbrev table used while in LLVM mode.")
(define-abbrev-table 'llvm-mode-abbrev-table ())

(defvar llvm-mode-hook nil)
(defvar llvm-mode-map nil)   ; Create a mode-specific keymap.

(if (not llvm-mode-map)
    ()  ; Do not change the keymap if it is already set up.
  (setq llvm-mode-map (make-sparse-keymap))
  (define-key llvm-mode-map "\t" 'tab-to-tab-stop)
  (define-key llvm-mode-map "\es" 'center-line)
  (define-key llvm-mode-map "\eS" 'center-paragraph))


(defun llvm-mode ()
  "Major mode for editing LLVM source files.
  \\{llvm-mode-map}
  Runs llvm-mode-hook on startup."
  (interactive)
  (kill-all-local-variables)
  (use-local-map llvm-mode-map)         ; Provides the local keymap.
  (setq major-mode 'llvm-mode)          

  (make-local-variable 'font-lock-defaults)
  (setq major-mode 'llvm-mode           ; This is how describe-mode
                                        ;   finds the doc string to print.
  mode-name "LLVM"                      ; This name goes into the modeline.
  font-lock-defaults `(llvm-font-lock-keywords))

  (setq local-abbrev-table llvm-mode-abbrev-table)
  (set-syntax-table llvm-mode-syntax-table)
  (setq comment-start ";")
  (run-hooks 'llvm-mode-hook))          ; Finally, this permits the user to
                                        ;   customize the mode with a hook.

;; Associate .ll files with llvm-mode
(setq auto-mode-alist
   (append '(("\\.ll$" . llvm-mode) ("\\.llx$" . llvm-mode)) auto-mode-alist))

(provide 'llvm-mode)
;; end of llvm-mode.el
