#include <string>

struct string_container {
  std::string short_string;
  std::string long_string;
  std::string *short_string_ptr = &short_string;
  std::string *long_string_ptr = &long_string;
  std::string &short_string_ref = short_string;
  std::string &long_string_ref = long_string;
};

int main()
{
    std::wstring wempty(L"");
    std::wstring s(L"hello world! מזל טוב!");
    std::wstring S(L"!!!!");
    const wchar_t *mazeltov = L"מזל טוב";
    std::string empty("");
    std::string q("hello world");
    std::string Q("quite a long std::strin with lots of info inside it");
    auto &rq = q, &rQ = Q;
    std::string *pq = &q, *pQ = &Q;
    string_container sc;
    sc.short_string = "u22";
    sc.long_string =
        "quite a long std::string with lots of info inside it inside a struct";

    S.assign(L"!!!!!"); // Set break point at this line.
    return 0;
}
