open Ocamlbuild_plugin;;

ocaml_lib ~extern:true "llvm";;
ocaml_lib ~extern:true "llvm_analysis";;

flag ["link"; "ocaml"; "g++"] (S[A"-cc"; A"g++"]);;
