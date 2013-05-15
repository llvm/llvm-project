#include <string>

int main()
{
    std::wstring s(L"hello world! מזל טוב!");
    std::wstring S(L"!!!!");
    const wchar_t *mazeltov = L"מזל טוב";
    std::string q("hello world");
    std::string Q("quite a long std::strin with lots of info inside it");
    S.assign(L"!!!!!"); // Set break point at this line.
    return 0;
}