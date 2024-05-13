#include <stdio.h>
#include <stdint.h>

namespace a {
    class c {
    public:
        c();
        ~c();
        void func1() 
        {
            puts (__PRETTY_FUNCTION__);
        }
        void func2() 
        {
            puts (__PRETTY_FUNCTION__);
        }
        void func3() 
        {
            puts (__PRETTY_FUNCTION__);
        }
    };

    c::c() {}
    c::~c() {}
}

namespace aa {
    class cc {
    public:
        cc();
        ~cc();
        void func1() 
        {
            puts (__PRETTY_FUNCTION__);
        }
        void func2() 
        {
            puts (__PRETTY_FUNCTION__);
        }
        void func3() 
        {
            puts (__PRETTY_FUNCTION__);
        }
    };

    cc::cc() {}
    cc::~cc() {}
}

namespace b {
    class c {
    public:
        c();
        ~c();
        void func1() 
        {
            puts (__PRETTY_FUNCTION__);
        }
        void func3() 
        {
            puts (__PRETTY_FUNCTION__);
        }
    };

    c::c() {}
    c::~c() {}
}

namespace c {
    class d {
    public:
        d () {}
        ~d() {}
        void func2() 
        {
            puts (__PRETTY_FUNCTION__);
        }
        void func3() 
        {
            puts (__PRETTY_FUNCTION__);
        }
    };
}

namespace ns {
template <typename Type> struct Foo {
  template <typename T> void import() {}

  template <typename T> auto func() {}

  operator bool() { return true; }

  template <typename T> operator T() { return {}; }

  template <typename T> void operator<<(T t) {}
};

template <typename Type> void g() {}
} // namespace ns

int main (int argc, char const *argv[])
{
    a::c ac;
    aa::cc aac;
    b::c bc;
    c::d cd;
    ac.func1();
    ac.func2();
    ac.func3();
    aac.func1();
    aac.func2();
    aac.func3();
    bc.func1();
    bc.func3();
    cd.func2();
    cd.func3();

    ns::Foo<double> f;
    f.import <int>();
    f.func<int>();
    f.func<ns::Foo<int>>();
    f.operator bool();
    f.operator a::c();
    f.operator ns::Foo<int>();
    f.operator<<(5);
    f.operator<< <ns::Foo<int>>({});

    ns::g<int>();
    ns::g<char>();

    return 0;
}
