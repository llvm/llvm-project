extern int call_through_indirect(int);
extern int reexport_to_indirect(int);

int
main ()
{
  int indirect_result = call_through_indirect(20); // Set breakpoint here to step in indirect.
  indirect_result = call_through_indirect(30);

  int reexport_result = reexport_to_indirect (20); // Set breakpoint here to step in reexported.
  reexport_result = reexport_to_indirect (30);

  return 0;
}
