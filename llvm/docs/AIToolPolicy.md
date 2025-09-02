# LLVM AI Tool Use Policy

LLVM's policy on AI-assisted tooling is fundamentally liberal -- We want to
enable contributors to use the latest and greatest tools available. Our policy
guided by two major concerns, the latter of which is the most important:

1.  Ensuring that contributions do not contain copyrighted content.
2.  Ensuring that contributions are not extractive and meet our
    [quality](#quality) bar.

This policy covers, but is not limited to, the following kinds of
contributions:

- Code, usually in the form of a pull request
- RFCs or design proposals
- Issues or security vulnerabilities
- Comments and feedback on pull requests

## Copyright

Artificial intelligence systems raise many questions around copyright that have
yet to be answered. Our policy on AI tools is similar to our copyright policy:
Contributors are responsible for ensuring that they have the right to
contribute code under the terms of our license, typically meaning that either
they, their employer, or their collaborators hold the copyright. Using AI tools
to regenerate copyrighted material does not remove the copyright, and
contributors are responsible for ensuring that such material does not appear in
their contributions. Contributions found to violate this policy will be removed
just like any other offending contribution.

## Quality

Sending patches, PRs, RFCs, comments, etc to LLVM, is not free -- it takes a
lot of maintainer time and energy to review those contributions! Recent
improvements in AI-assisted tooling have made it easy to generate large volumes
of code and text with little effort on the part of the contributor. This has
increased the asymmetry between the work of producing a contribution, and the
work of reviewing the contribution. Our **golden rule** is that a contribution
should be worth more to the project than the time it takes to review it. These
ideas are captured by this quote from the book [Working in Public][1] by
Nadia Eghbal:

[1]: https://press.stripe.com/working-in-public

> \"When attention is being appropriated, producers need to weigh the costs and
> benefits of the transaction. To assess whether the appropriation of attention
> is net-positive, it's useful to distinguish between *extractive* and
> *non-extractive* contributions. Extractive contributions are those where the
> marginal cost of reviewing and merging that contribution is greater than the
> marginal benefit to the project's producers. In the case of a code
> contribution, it might be a pull request that's too complex or unwieldy to
> review, given the potential upside.\" \-- Nadia Eghbal

We encourage contributions that help sustain the project. We want the LLVM
project to be welcoming and open to aspiring compiler engineers who are willing
to invest time and effort to learn and grow, because growing our contributor
base and recruiting new maintainers helps sustain the project over the long
term. We therefore automatically post a greeting comment to pull requests from
new contributors and encourage maintainers to spend their time to help new
contributors learn.

However, we expect to see a growth pattern in the quality of a contributor's
work over time. Maintainers are empowered to push back against *extractive*
contributions and explain why they believe a contribution is overly burdensome
or not aligned with the project goals.

If a maintainer judges that a contribution is extractive (i.e. it is generated
with tool-assistance or isn't valuable for other reasons), they should
copy-paste the following response, add the `extractive` label if applicable,
and refrain from further engagement:

    This PR appears to be extractive, and requires additional justification for
    why it is valuable enough to the project for us to review it. Please see
    our developer policy on AI-generated contributions:
    http://llvm.org/docs/AIToolPolicy.html

Other reviewers should use the label prioritize their review time.

Contributors are welcome to improve their work or make the case for why it has
value for the community, but they should keep in mind that they may be
moderated for excessive extractive communications.

While our quality policy is subjective at its core, here are some guidelines
that can be used to assess the quality of a contribution:

- Contribution size: Larger contributions require more time to read and review.
  RFCs and issues should be clear and concise, and pull requests should not
  change unrelated code.
- Potential user base: Contributions with more users are inherently more valuable.
- Code must adhere to the [LLVM Coding Standards](CodingStandards.html).
- Pull requests should build and pass premerge checks. For first-time
  contributors, this will require an initial cursory review to run the
  checks.

The best ways to make a change less extractive and more valuable are to reduce
its size or complexity or to increase its usefulness to the community. These
factors are impossible to weigh objectively, and our project policy leaves this
determination up to the maintainers of the project, i.e. those who are doing
the work of sustaining the project.

We encourage, but do not require, contributors making large changes to document
the tools that they used as part of the rationale for why they believe their
contribution has merit. This is similar in spirit to including a sed or Python
script in the commit message when making large-scale changes to the project,
such as updating the LLVM IR textual syntax.

## Examples

Here are some examples of contributions that demonstrate how to apply
the principles of this policy:

- [This PR](https://github.com/llvm/llvm-project/pull/142869) contains a
  proof from Alive2, which is a strong signal of value and correctness.
- This [generated
  documentation](https://discourse.llvm.org/t/searching-for-gsym-documentation/85185/2)
  was reviewed for correctness by a human before being posted.

## References

Our policy was informed by experiences in other communities:

- [Rust policy on burdensome
  PRs](https://github.com/rust-lang/compiler-team/issues/893)
- [Seth Larson's post](https://sethmlarson.dev/slop-security-reports)
  on slop security reports in the Python ecosystem
- The METR paper [Measuring the Impact of Early-2025 AI on Experienced
  Open-Source Developer
  Productivity](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/).
- [QEMU bans use of AI content
  generators](https://www.qemu.org/docs/master/devel/code-provenance.html#use-of-ai-content-generators)
