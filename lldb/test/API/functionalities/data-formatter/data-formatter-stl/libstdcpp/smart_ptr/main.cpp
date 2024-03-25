#include <memory>
#include <string>

struct Foo {
  int mem = 5;
};

int
main()
{
    std::shared_ptr<char> nsp;
    std::shared_ptr<int> isp(new int{123});
    std::shared_ptr<std::string> ssp = std::make_shared<std::string>("foobar");
    std::shared_ptr<Foo> fsp = std::make_shared<Foo>();

    std::weak_ptr<char> nwp;
    std::weak_ptr<int> iwp = isp;
    std::weak_ptr<std::string> swp = ssp;

    nsp.reset(); // Set break point at this line.
    isp.reset();
    ssp.reset();
    fsp.reset();

    return 0; // Set break point at this line.
}
