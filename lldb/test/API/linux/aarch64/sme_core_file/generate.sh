run () {
  ./a.out "$@"
  mv core core_$(echo "$*" | sed 's/ /_/g')
}

run 0 16 32 1
run 0 32 16 0
run 1 16 32 0
run 1 32 16 1
