;;; tablegen-mode.el --- Major mode for LLVM TableGen description files -*- lexical-binding: t -*-

;; Maintainer:  The LLVM team, http://llvm.org/
;; Version: 2.0
;; Homepage: http://llvm.org/
;; Package-Requires: ((emacs "24.3"))

;;; Commentary:
;; A major mode for TableGen description files in LLVM.

(require 'comint)
(require 'custom)
(require 'ansi-color)

;; Create mode-specific tables.
;;; Code:

(defface tablegen-decorators-face
  '((t :inherit font-lock-function-call-face))
  "Face method decorators.")

(defvar tablegen-font-lock-keywords
  (let ((kw (regexp-opt '("class" "defm" "def" "field" "include" "in"
                          "let" "multiclass" "foreach" "if" "then" "else"
                          "defvar" "defset" "dump" "assert")
                        'words))
        (type-kw (regexp-opt '("bit" "bits" "code" "dag" "int" "list" "string")
                             'words)))
    `(
      ;; Strings
      ("\"[^\"]+\"" . font-lock-string-face)
      ;; Hex constants
      ("\\<0x[0-9A-Fa-f]+\\>" . font-lock-preprocessor-face)
      ;; Binary constants
      ("\\<0b[01]+\\>" . font-lock-preprocessor-face)
      ;; Integer literals
      ("\\<[-]?[0-9]+\\>" . font-lock-preprocessor-face)
      ;; Floating point constants
      ("\\<[-+]?[0-9]+\.[0-9]*\([eE][-+]?[0-9]+\)?\\>" . font-lock-preprocessor-face)

      ("^[ \t]*\\(@.+\\)" 1 'tablegen-decorators-face)
      ;; Operators
      ("\\![a-zA-Z]+" . font-lock-function-name-face)
      ;; Keywords
      (,kw . font-lock-keyword-face)
      ;; Type keywords
      (,type-kw . font-lock-type-face)))
  "Additional expressions to highlight in TableGen mode.")

;; ---------------------- Syntax table ---------------------------

(defvar tablegen-mode-syntax-table nil
  "Syntax table used in `tablegen-mode' buffers.")
(when (not tablegen-mode-syntax-table)
  (setq tablegen-mode-syntax-table (make-syntax-table))
  ;; whitespace (` ')
  (modify-syntax-entry ?\   " "      tablegen-mode-syntax-table)
  (modify-syntax-entry ?\t  " "      tablegen-mode-syntax-table)
  (modify-syntax-entry ?\r  " "      tablegen-mode-syntax-table)
  (modify-syntax-entry ?\n  " "      tablegen-mode-syntax-table)
  (modify-syntax-entry ?\f  " "      tablegen-mode-syntax-table)
  ;; word constituents (`w')
  (modify-syntax-entry ?\%  "w"      tablegen-mode-syntax-table)
  (modify-syntax-entry ?\_  "w"      tablegen-mode-syntax-table)
  ;; comments
  (modify-syntax-entry ?/   ". 124b" tablegen-mode-syntax-table)
  (modify-syntax-entry ?*   ". 23"   tablegen-mode-syntax-table)
  (modify-syntax-entry ?\n  "> b"    tablegen-mode-syntax-table)
  ;; open paren (`(')
  (modify-syntax-entry ?\(  "()"      tablegen-mode-syntax-table)
  (modify-syntax-entry ?\[  "(]"      tablegen-mode-syntax-table)
  (modify-syntax-entry ?\{  "(}"      tablegen-mode-syntax-table)
  (modify-syntax-entry ?\<  "(>"      tablegen-mode-syntax-table)
  ;; close paren (`)')
  (modify-syntax-entry ?\)  ")("      tablegen-mode-syntax-table)
  (modify-syntax-entry ?\]  ")["      tablegen-mode-syntax-table)
  (modify-syntax-entry ?\}  "){"      tablegen-mode-syntax-table)
  (modify-syntax-entry ?\>  ")<"      tablegen-mode-syntax-table)
  ;; string quote ('"')
  (modify-syntax-entry ?\"  "\"\""     tablegen-mode-syntax-table))

;; --------------------- Abbrev table -----------------------------

(defvar tablegen-mode-abbrev-table nil
  "Abbrev table used while in TableGen mode.")
(define-abbrev-table 'tablegen-mode-abbrev-table ())

(defvar tablegen-mode-hook nil)
(defvar tablegen-mode-map nil)   ; Create a mode-specific keymap.

(unless tablegen-mode-map
  (setq tablegen-mode-map (make-sparse-keymap))
  (define-key tablegen-mode-map "\t"  'tab-to-tab-stop)
  (define-key tablegen-mode-map "\es" 'center-line)
  (define-key tablegen-mode-map "\eS" 'center-paragraph))

;;;###autoload
(define-derived-mode tablegen-mode prog-mode "TableGen"
  "Major mode for editing TableGen description files."
  (setq font-lock-defaults `(tablegen-font-lock-keywords))
  (setq-local require-final-newline t)
  (setq-local comment-start "//")
  (setq-local indent-tabs-mode nil))

;; Associate .td files with tablegen-mode
;;;###autoload
(add-to-list 'auto-mode-alist (cons (purecopy "\\.td\\'")  'tablegen-mode))

(provide 'tablegen-mode)

;;; tablegen-mode.el ends here
