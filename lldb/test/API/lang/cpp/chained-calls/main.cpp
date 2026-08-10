class Bool {
public:
    Bool operator&(const Bool other)
    {
        Bool result;
        result.value = value && other.value;
        return result;
    }

    bool value;
};

bool get(Bool object)
{
    return object.value;
}

Bool set(bool value)
{
    Bool result;
    result.value = value;
    return result;
}

int main()
{
    Bool t = set(true);
    Bool f = set(false);
    get(t);
    get(f);
    get(t & f);
    return 0; // break here
}
