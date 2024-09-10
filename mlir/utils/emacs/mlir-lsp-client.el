;;; mlir-lsp-clinet.el --- LSP clinet for the MLIR.

;; Copyright (C) 2022 The MLIR Authors.
;;
;; Licensed under the Apache License, Version 2.0 (the "License");
;; you may not use this file except in compliance with the License.
;; You may obtain a copy of the License at
;;
;;      http://www.apache.org/licenses/LICENSE-2.0
;;
;; Unless required by applicable law or agreed to in writing, software
;; distributed under the License is distributed on an "AS IS" BASIS,
;; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
;; See the License for the specific language governing permissions and
;; limitations under the License.

;; Version: 0.1.0

;;; Commentary:

;; LSP clinet to use with `mlir-mode' that uses `mlir-lsp-server' or any
;; user made compatible server.

;;; Code:
(require 'lsp-mode)

(defgroup lsp-mlir nil
  "LSP support for MLIR."
  :group 'lsp-mode
  :link '(url-link "https://mlir.llvm.org/docs/Tools/MLIRLSP/"))


(defcustom lsp-mlir-server-executable "mlir-lsp-server"
  "Command to start the mlir language server."
  :group 'lsp-mlir
  :risky t
  :type 'file)


(defun lsp-mlir-setup ()
  "Setup the LSP client for MLIR."
  (add-to-list 'lsp-language-id-configuration '(mlir-mode . "mlir"))

  (lsp-register-client
   (make-lsp-client
    :new-connection (lsp-stdio-connection (lambda () lsp-mlir-server-executable))
    :activation-fn (lsp-activate-on "mlir")
    :priority -1
    :server-id 'mlir-lsp)))


(provide 'mlir-lsp-client)
;;; mlir-lsp-client.el ends here
