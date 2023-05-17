#include <string>

int main()
{
    std::wstring wempty(L"");
    std::wstring s(L"hello world! מזל טוב!");
    std::wstring S(L"!!!!");
    const wchar_t *mazeltov = L"מזל טוב";
    std::string empty("");
    std::string q("hello world");
    std::string Q("quite a long std::strin with lots of info inside it");
    std::basic_string<unsigned char> uchar(5, 'a');
    auto &rq = q, &rQ = Q;
    std::string *pq = &q, *pQ = &Q;
    S.assign(L"!!!!!"); // Set break point at this line.
    return 0;
}
