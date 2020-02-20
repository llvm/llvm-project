int string_not_empty (const char *s)
{
    if (s && s[0])
        return 1;
    return 0;
}

int main (int argc, char const *argv[])
{
    int (*callback)(const char *) = string_not_empty;

    return callback(0); // Set break point at this line.
}
