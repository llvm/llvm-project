// RUN: %llvmgxx -S %s -o -

struct PrefMapElem {
  virtual ~PrefMapElem(); 
  unsigned int fPrefId;
};

int foo() {
  PrefMapElem* fMap;
  if (fMap[0].fPrefId == 1)
    return 1;
  
  return 0;
}
