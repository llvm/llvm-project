#!/bin/sh -e

scriptpath=$(which "$0")
scriptpath=$(readlink -f "$scriptpath")
bindir=$(dirname "$scriptpath")
prefix=$(dirname "$bindir")

cmd="$1"

case "$cmd" in
build | get | install | run | test)
  shift
  PATH="$prefix/lib/llgo/go-path:$PATH" exec go "$cmd" -compiler gccgo "$@"
  ;;

*)
  exec go "$@"
  ;;
esac
