@import Darwin;
@import myModule;

int main()
{
    int a = isInline(2);
    int b = notInline();
    printf("%d %d\n", a, b); // Set breakpoint here.
}
