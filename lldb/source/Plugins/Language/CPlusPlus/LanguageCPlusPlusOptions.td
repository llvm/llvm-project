include "../../../../source/Commands/OptionsBase.td"

let Command = "expression cplusplus" in {
  def ignore_context_qualifiers
      : Option<"c++-ignore-context-qualifiers", "Q">,
        Groups<[1, 2]>,
        Desc<"When specified, evaluates the expression without taking into "
             "account the type ${Q}ualifiers of the scope. In C++, this would "
             "permit calling a non-const method when stopped in a const-method "
             "(which would be disallowed by language rules).">;
}
