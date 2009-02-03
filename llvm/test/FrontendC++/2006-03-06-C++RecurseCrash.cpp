// RUN: %llvmgcc %s -S -o -
namespace std {
  class exception { };

  class type_info {
  public:
    virtual ~type_info();
  };

}

namespace __cxxabiv1 {
  class __si_class_type_info : public std::type_info {
    ~__si_class_type_info();
  };
}

class recursive_init: public std::exception {
public:
  virtual ~recursive_init() throw ();
};

recursive_init::~recursive_init() throw() { }

