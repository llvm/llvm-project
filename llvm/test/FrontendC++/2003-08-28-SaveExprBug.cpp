// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null


char* eback();

template<typename foo>
struct basic_filebuf {
  char *instancevar;

  void callee() {
    instancevar += eback() != eback();
  }

  void caller();
};


template<typename _CharT>
void basic_filebuf<_CharT>::caller() {
  callee();
}


template class basic_filebuf<char>;
